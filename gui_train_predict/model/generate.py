#!/usr/bin/env python
import os
import sys
import numpy as np
from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, concatenate, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import *
from keras.models import model_from_json
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape, Dense
from keras.models import Sequential, Model
import cv2

#sampler

#vocab
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "
SEPARATOR = '->'

class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}
        self.token_lookup = {}
        self.size = 0

        self.append(START_TOKEN)
        self.append(END_TOKEN)
        self.append(PLACEHOLDER)

    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1

    def create_binary_representation(self):
        if sys.version_info >= (3,):
            items = self.vocabulary.items()
        else:
            items = self.vocabulary.iteritems()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary

    def get_serialized_binary_representation(self):
        if len(self.binary_vocabulary) == 0:
            self.create_binary_representation()

        string = ""
        if sys.version_info >= (3,):
            items = self.binary_vocabulary.items()
        else:
            items = self.binary_vocabulary.iteritems()
        for key, value in items:
            array_as_string = np.array2string(value, separator=',', max_line_width=self.size * self.size)
            string += "{}{}{}\n".format(key, SEPARATOR, array_as_string[1:len(array_as_string) - 1])
        return string

    def save(self, path):
        output_file_name = "{}/words.vocab".format(path)
        output_file = open(output_file_name, 'w')
        output_file.write(self.get_serialized_binary_representation())
        output_file.close()

    def retrieve(self, path):
        input_file = open("{}/words.vocab".format(path), 'r')
        buffer = ""
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(SEPARATOR)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(SEPARATOR):]
                value = np.fromstring(value, sep=',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = np.where(value == 1)[0][0]
                self.token_lookup[np.where(value == 1)[0][0]] = key

                buffer = ""
            except ValueError:
                buffer += line
        input_file.close()
        self.size = len(self.vocabulary)
        
#beam and nodes

class Node:
    def __init__(self, key, value, data=None):
        self.key = key
        self.value = value
        self.data = data
        self.parent = None
        self.root = None
        self.children = []
        self.level = 0

    def add_children(self, children, beam_width):
        for child in children:
            child.level = self.level + 1
            child.value = child.value * self.value

        nodes = sorted(children, key=lambda node: node.value, reverse=True)
        nodes = nodes[:beam_width]

        for node in nodes:
            self.children.append(node)
            node.parent = self

        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root
        child.root = self.root

    def remove_child(self, child):
        self.children.remove(child)

    def max_child(self):
        if len(self.children) == 0:
            return self

        max_childs = []
        for child in self.children:
            max_childs.append(child.max_child())

        nodes = sorted(max_childs, key=lambda child: child.value, reverse=True)
        return nodes[0]

    def show(self, depth=0):
        print(" " * depth, self.key, self.value, self.level)
        for child in self.children:
            child.show(depth + 2)


class BeamSearch:
    def __init__(self, beam_width=1):
        self.beam_width = beam_width

        self.root = None
        self.clear()

    def search(self):
        result = self.root.max_child()

        self.clear()
        return self.retrieve_path(result)

    def add_nodes(self, parent, children):
        parent.add_children(children, self.beam_width)

    def is_valid(self):
        leaves = self.get_leaves()
        level = leaves[0].level
        counter = 0
        for leaf in leaves:
            if leaf.level == level:
                counter += 1
            else:
                break

        if counter == len(leaves):
            return True

        return False

    def get_leaves(self):
        leaves = []
        self.search_leaves(self.root, leaves)
        return leaves

    def search_leaves(self, node, leaves):
        for child in node.children:
            if len(child.children) == 0:
                leaves.append(child)
            else:
                self.search_leaves(child, leaves)

    def prune_leaves(self):
        leaves = self.get_leaves()

        nodes = sorted(leaves, key=lambda leaf: leaf.value, reverse=True)
        nodes = nodes[self.beam_width:]

        for node in nodes:
            node.parent.remove_child(node)

        while not self.is_valid():
            leaves = self.get_leaves()
            max_level = 0
            for leaf in leaves:
                if leaf.level > max_level:
                    max_level = leaf.level

            for leaf in leaves:
                if leaf.level < max_level:
                    leaf.parent.remove_child(leaf)

    def clear(self):
        self.root = None
        self.root = Node("root", 1.0, None)

    def retrieve_path(self, end):
        path = [end.key]
        data = [end.data]
        while end.parent is not None:
            end = end.parent
            path.append(end.key)
            data.append(end.data)

        result_path = []
        result_data = []
        for i in range(len(path) - 2, -1, -1):
            result_path.append(path[i])
            result_data.append(data[i])
        return result_path, result_data

#utils
class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

class Sampler:
    def __init__(self, voc_path, input_shape, output_size, context_length):
        self.voc = Vocabulary()
        self.voc.retrieve(voc_path)

        self.input_shape = input_shape
        self.output_size = output_size

        print("Vocabulary size: {}".format(self.voc.size))
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        self.context_length = context_length

    def predict_greedy(self, model, input_img, require_sparse_label=True, sequence_length=150, verbose=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            probas = model.predict(input_img, np.array([current_context]))
            prediction = np.argmax(probas)
            out_probas.append(probas)

            new_context = []
            for j in range(1, self.context_length):
                new_context.append(current_context[j])

            if require_sparse_label:
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)
            else:
                new_context.append(prediction)

            current_context = new_context

            predictions += self.voc.token_lookup[prediction]

            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions, out_probas

    def recursive_beam_search(self, model, input_img, current_context, beam, current_node, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))

        predictions = []
        for i in range(0, len(probas)):
            predictions.append((i, probas[i], probas))

        nodes = []
        for i in range(0, len(predictions)):
            prediction = predictions[i][0]
            score = predictions[i][1]
            output_probas = predictions[i][2]
            nodes.append(Node(prediction, score, output_probas))

        beam.add_nodes(current_node, nodes)

        if beam.is_valid():
            beam.prune_leaves()
            if sequence_length == 1 or self.voc.token_lookup[beam.root.max_child().key] == END_TOKEN:
                return

            for node in beam.get_leaves():
                prediction = node.key

                new_context = []
                for j in range(1, self.context_length):
                    new_context.append(current_context[j])
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)

                self.recursive_beam_search(model, input_img, new_context, beam, node, sequence_length - 1)

    def predict_beam_search(self, model, input_img, beam_width=3, require_sparse_label=True, sequence_length=150):
        predictions = START_TOKEN
        out_probas = []

        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Utils.sparsify(current_context, self.output_size)

        beam = BeamSearch(beam_width=beam_width)

        self.recursive_beam_search(model, input_img, current_context, beam, beam.root, sequence_length)

        predicted_sequence, probas_sequence = beam.search()

        for k in range(0, len(predicted_sequence)):
            prediction = predicted_sequence[k]
            probas = probas_sequence[k]
            out_probas.append(probas)

            predictions += self.voc.token_lookup[prediction]

        return predictions, out_probas


#config
CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000

#AModel
class AModel:
    def __init__(self, input_shape, output_size, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        self.name = ""

    def save(self):
        model_json = self.model.to_json()
        with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))

    def load(self, name=""):
        output_name = self.name if name == "" else name
        with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        print("!!")
        print(self.model)
        print(output_name)
        self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))
        print("$$")
        
#autoencoder_image

class autoencoder_image(AModel):
	def __init__(self, input_shape, output_size, output_path):
		AModel.__init__(self, input_shape, output_size, output_path)
		self.name = 'autoencoder'

		input_image = Input(shape=input_shape)
		encoder = Conv2D(32, 3, padding='same', activation='relu')(input_image)
		encoder = Conv2D(32, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoder = Dropout(0.25)(encoder)

		encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)
		encoder = Conv2D(64, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoder = Dropout(0.25)(encoder)

		encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)
		encoder = Conv2D(128, 3, padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D()(encoder)
		encoded = Dropout(0.25, name='encoded_layer')(encoder)

		decoder = Conv2DTranspose(128, 3, padding='same', activation='relu')(encoded)
		decoder = Conv2DTranspose(128, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoder = Dropout(0.25)(decoder)

		decoder = Conv2DTranspose(64, 3, padding='same', activation='relu')(decoder)
		decoder = Conv2DTranspose(64, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoder = Dropout(0.25)(decoder)

		decoder = Conv2DTranspose(32, 3, padding='same', activation='relu')(decoder)
		decoder = Conv2DTranspose(3, 3, padding='same', activation='relu')(decoder)
		decoder = UpSampling2D()(decoder)
		decoded = Dropout(0.25)(decoder)

		# decoder = Dense(256*256*3)(decoder)
		# decoded = Reshape(target_shape=input_shape)(decoder)

		self.model = Model(input_image, decoded)
		self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
		self.model.summary()

	def fit_generator(self, generator, steps_per_epoch):
		self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
		self.save()

	def predict_hidden(self, images):
		hidden_layer_model = Model(inputs = self.input, outputs = self.get_layer('encoded_layer').output)
		return hidden_layer_model.predict(images)

class guicode(AModel):
	def __init__(self, input_shape, output_size, output_path):
		AModel.__init__(self, input_shape, output_size, output_path)
		self.name = "guicode"

		visual_input = Input(shape=input_shape)

		#Load the pre-trained autoencoder model
		autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
		autoencoder_model.load('autoencoder')
		autoencoder_model.model.load_weights('../bin/autoencoder.h5')

		#Get only the model up to the encoded part
		hidden_layer_model_freeze = Model(inputs=autoencoder_model.model.input, outputs=autoencoder_model.model.get_layer('encoded_layer').output)
		hidden_layer_input = hidden_layer_model_freeze(visual_input)
		
		#Additional layers before concatenation
		hidden_layer_model = Flatten()(hidden_layer_input)
		hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
		hidden_layer_model = Dropout(0.3)(hidden_layer_model)
		hidden_layer_model = Dense(1024, activation='relu')(hidden_layer_model)
		hidden_layer_model = Dropout(0.3)(hidden_layer_model)
		hidden_layer_result = RepeatVector(CONTEXT_LENGTH)(hidden_layer_model)

		#Make sure the loaded hidden_layer_model_freeze will no longer be updated
		for layer in hidden_layer_model_freeze.layers:
			layer.trainable = False

		language_model = Sequential()
		language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
		language_model.add(LSTM(128, return_sequences=True))

		textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
		encoded_text = language_model(textual_input)

		decoder = concatenate([hidden_layer_result, encoded_text])

		decoder = LSTM(512, return_sequences=True)(decoder)
		decoder = LSTM(512, return_sequences=False)(decoder)
		decoder = Dense(output_size, activation='softmax')(decoder)

		self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

		optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	def fit_generator(self, generator, steps_per_epoch):
		self.model.summary()
		self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
		self.save()

	def predict(self, image, partial_caption):
		return self.model.predict([image, partial_caption], verbose=0)[0]

	def predict_batch(self, images, partial_captions):
		return self.model.predict([images, partial_captions], verbose=1)




argv = sys.argv[1:]

if len(argv) < 4:
    print("Error: not enough argument supplied:")
    print("generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]
    output_path = argv[3]
    search_method = "greedy" if len(argv) < 5 else argv[4]

print(trained_weights_path)
print(trained_model_name)
print("{}/meta_dataset.npy".format(trained_weights_path))
meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path),allow_pickle=True)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]


model = guicode(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

for f in os.listdir(input_path):
    if f.find(".png") != -1:
        evaluation_img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)

        file_name = f[:f.find(".png")]

        if search_method == "greedy":
            result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
            print("Result greedy: {}".format(result))
        else:
            beam_width = int(search_method)
            print("Search with beam width: {}".format(beam_width))
            result, _ = sampler.predict_beam_search(model, np.array([evaluation_img]), beam_width=beam_width)
            print("Result beam: {}".format(result))

        with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
            out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))
