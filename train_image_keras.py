#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:09:47 2018

@author: atif

This file will use for train Image using Keras

"""

#####################
#Importing library###
#####################

import numpy as np
from skimage import io, color, exposure, transform
from skimage.color import rgb2gray
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split  #it came from update scikit learn. https://stackoverflow.com/questions/40704484/importerror-no-module-named-model-selection
import os
import glob
import h5py
import json

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.pooling import MaxPooling2D

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')

from matplotlib import pyplot as plt

# import json file where all variable values are stored. please take a look to modify it.

with open('variable_config.json', 'r') as f:
    config = json.load(f)


NUM_CLASSES = config['DEFAULT']['num_class']
IMG_SIZE = config['DEFAULT']['img_size']
number_filter = config['DEFAULT']['number_filter']
filter_size = config['DEFAULT']['filter_size']
img_depth = config['DEFAULT']['img_depth']
img_type = config['DEFAULT']['img_type']
epochs = config['DEFAULT']['epochs']
batch_size = config['DEFAULT']['batch_size']
train_image_path = config['DEFAULT']['train_image_path']
test_image_path = config['DEFAULT']['test_image_path']
learning_model_path = config['DEFAULT']['learning_model_path']
validation_split = config['DEFAULT']['validation_split']

'''
NUM_CLASSES = 9
IMG_SIZE = 48
number_filter=5
'''
###################################
#function for Preprocessing Image##
###################################

#for gray scale
def preprocess_img(img):
    # Histogram normalization in y
#     hsv = color.rgb2hsv(img)
#     hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
#     img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    img = rgb2gray(img)

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])

###############################################################
#Storing train images and their label in arrays################
###############################################################
    
imgs = []
labels = []
root_dir = train_image_path
#root_dir = '/home/atif/training_by_several_learning_process/number_classify/rgb_2_gray/Image-classification/train_image/'
#path='/home/atif/training_by_several_learning_process/flower_photos/00000/'

#all_img_paths = glob.glob(path+ '5547758_eea9edfd54_n_000.jpg')

all_img_paths = glob.glob(os.path.join(root_dir, '*/*'+str(img_type))) #I have done the training with .ppm format image. If another type of image will come 
                                                                                    #them .ppm will be changed by that extension
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    try:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

        if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            #print("get it 2")
    except (IOError, OSError):
        print('missed', img_path)
        pass

X = np.array(imgs, dtype='float32') #Keeping the image as an array
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels] #labels of the image


##################################################################
########### Reshaping ############################################
##################################################################

X = np.array(imgs, dtype='float32')
print(X.shape)
# plt.imshow(X[0])
# plt.imshow(X[0],cmap="gray")
plt.imshow(X[0]) #if you use this command here you will see something coloured image. No problem, it is gray image. 
                        #To see full Black and white image uncomment the previous line.
X = X.reshape(len(imgs),1,IMG_SIZE,IMG_SIZE) # write (IMG_SIZE,IMG_SIZE,1 if you want channel last; 1= grayscale;3=RGB)
# plt.imshow(X[0],cmap="gray")
print(X.shape)
print(X.ndim)
print(X[0].shape)

print(X.shape)
print(Y.shape)

################################################################
################## Declare the Model ###########################
################################################################

# This model is for understanding the inner calculation of CNN that's why I have started witha minimal layer as well as model.
# Increase the filter number and layer if you want a good result
#Conv2D(1, (3, 3) >> here 1 = number of filter. (3,3) = kernel height and width
# you can just add padding just beside Conv2D. (model.add(Conv2D(1,(3,3)),padding='same',....))
# I haven't added here for remove complexity in c++(I have tried to implement this whole model in testing phase in cpp)
# def cnn_model():
# #      padding='same'
#     model = Sequential()
#
#     model.add(Conv2D(number_filter, (filter_size, filter_size),
#                      input_shape=(img_depth,IMG_SIZE, IMG_SIZE),
#                      activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(NUM_CLASSES, activation='softmax'))
#
#     return model


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(img_depth, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #    model.add(Dense(2048, activation='relu'))
    #    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

model = cnn_model()

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


#################################################################
########### Display model summary, not necessary ################
#################################################################

model.summary()


################################################################
############### Training phase #################################
################################################################

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size = batch_size
epochs = epochs
model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split,
          #np.resize(img, (-1, <image shape>)
          callbacks=[LearningRateScheduler(lr_schedule),ModelCheckpoint(learning_model_path+'learning_model.h5', save_best_only=True)])
