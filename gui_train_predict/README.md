## Code:

# Link for dataset :
https://github.com/tonybeltramelli/pix2code/raw/master/datasets/pix2code_datasets.zip
unzip dataset and paste it into dataset folder

# split training set and evaluation set [no training example in the evaluation set]
cd ../model
python build_datasets.py ../datasets/web/all_data

# transform images (normalized pixel values and resized pictures) in training dataset to numpy arrays
python convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features

# provide input path to training data and output path to save trained model and metadata
python train.py ../datasets/web/training_set ../bin

# train on images pre-processed as arrays
python train.py ../datasets/web/training_features ../bin

# train with autoencoder
python train.py ../datasets/web/training_features ../bin 1

# Link for the pretrained Model :
https://drive.google.com/file/d/1BUgCYFG6aYWF-vZO2mcvFpY3l2YQGc3n/view?usp=sharing

Paste the pretrained model under bin folder.

# generate .gui file :
python generate.py ../bin guicode ../datasets/web/eval_set ../code

# Generate code for a single GUI image:
python sample.py ../bin guicode ../test_gui.png ../code

# Compile generated code to target language:
cd compiler
python web-compiler.py <input file path>.gui