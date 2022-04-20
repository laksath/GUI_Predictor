#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, concatenate, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import *
from keras.models import model_from_json
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape, Dense
from keras.models import Sequential, Model
from keras.backend import clear_session

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#Generator

#Dataset

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

#Utils
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
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")


#Config
CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000

class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.voc = Vocabulary()
        self.size = 0

    @staticmethod
    def load_paths_only(path):
        print("Parsing data...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load(self, path, generate_binary_sequences=False):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
                    self.append(file_name, gui, img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)

        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.next_words = self.sparsify_labels(self.next_words, self.voc)
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(",", " ,").replace("\n", " \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
        token_sequence.append(END_TOKEN)

        suffix = [PLACEHOLDER] * CONTEXT_LENGTH

        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

        return temp

    def save_metadata(self, path):
        np.save("{}/meta_dataset".format(path), np.array([self.input_shape, self.output_size, self.size]))


class Generator:
    @staticmethod
    def data_generator(voc, gui_paths, img_paths, batch_size, input_shape, generate_binary_sequences=False, verbose=False, loop_only_one=False, images_only=False):
        assert len(gui_paths) == len(img_paths)
        voc.create_binary_representation()

        while 1:
            batch_input_images = []
            batch_partial_sequences = []
            batch_next_words = []
            sample_in_batch_counter = 0

            for i in range(0, len(gui_paths)):
                if img_paths[i].find(".png") != -1:
                    img = Utils.get_preprocessed_img(img_paths[i], IMAGE_SIZE)
                else:
                    img = np.load(img_paths[i])["features"]
                gui = open(gui_paths[i], 'r')

                token_sequence = [START_TOKEN]
                for line in gui:
                    line = line.replace(",", " ,").replace("\n", " \n")
                    tokens = line.split(" ")
                    for token in tokens:
                        voc.append(token)
                        token_sequence.append(token)
                token_sequence.append(END_TOKEN)

                suffix = [PLACEHOLDER] * CONTEXT_LENGTH

                a = np.concatenate([suffix, token_sequence])
                for j in range(0, len(a) - CONTEXT_LENGTH):
                    context = a[j:j + CONTEXT_LENGTH]
                    label = a[j + CONTEXT_LENGTH]

                    batch_input_images.append(img)
                    batch_partial_sequences.append(context)
                    batch_next_words.append(label)
                    sample_in_batch_counter += 1

                    if sample_in_batch_counter == batch_size or (loop_only_one and i == len(gui_paths) - 1):
                        if verbose:
                            print("Generating sparse vectors...")
                        batch_next_words = Dataset.sparsify_labels(batch_next_words, voc)
                        if generate_binary_sequences:
                            batch_partial_sequences = Dataset.binarize(batch_partial_sequences, voc)
                        else:
                            batch_partial_sequences = Dataset.indexify(batch_partial_sequences, voc)

                        if verbose:
                            print("Convert arrays...")
                        batch_input_images = np.array(batch_input_images)
                        batch_partial_sequences = np.array(batch_partial_sequences)
                        batch_next_words = np.array(batch_next_words)

                        if verbose:
                            print("Yield batch")
						#include a generator for images only for autoencoder
                        if images_only:
                            yield(batch_input_images, batch_input_images)
                        else:
                            yield ([batch_input_images, batch_partial_sequences], batch_next_words)

                        batch_input_images = []
                        batch_partial_sequences = []
                        batch_next_words = []
                        sample_in_batch_counter = 0



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





#removed some training parameters, so as to make the code input easier
def run(input_path, output_path, train_autoencoder=False):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    gui_paths, img_paths = Dataset.load_paths_only(input_path)

    input_shape = dataset.input_shape
    output_size = dataset.output_size
    steps_per_epoch = dataset.size / BATCH_SIZE

    voc = Vocabulary()
    voc.retrieve(output_path)

	
    generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, input_shape=input_shape, generate_binary_sequences=True)
	
	#Included a generator for images only as an input for autoencoders
    generator_images = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, input_shape=input_shape, generate_binary_sequences=True, images_only=True)

	#For training of autoencoders 
    if train_autoencoder:
        autoencoder_model = autoencoder_image(input_shape, input_shape, output_path)
        autoencoder_model.fit_generator(generator_images, steps_per_epoch=steps_per_epoch)
        clear_session()
    
    model = guicode(input_shape, output_size, output_path)
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <train_autoencoder default: 0>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        train_autoencoder = False if len(argv) < 3 else True if int(argv[2]) == 1 else False

    run(input_path, output_path, train_autoencoder=train_autoencoder)
