# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:37:23 2021

@author: Angad
"""

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if(gpus):
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
import keras

from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras_vggface.vggface import VGGFace

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))

# base_model = keras.applications.MobileNetV2(
#     weights='imagenet',
#     input_shape=(160,160, 3),
#     include_top=False)
# Freeze base model
for layer in vgg_model.layers:
    layer.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = vgg_model.output
x = GlobalAveragePooling2D()(x)

x2 = Dense(512, activation='relu')(x)

x = Dropout(0.03)(x2)

x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

predictions = Dense(136)(x)

# add your top layer block to your base model
model = Model(vgg_model.input, predictions)
print(model.summary())


model.load_weights("training_9_518.ckpt")

import os
import pandas
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

files_path = [os.path.relpath(x) for x in os.listdir('./data/training/')]
csv = pandas.read_csv('./data/training_frames_keypoints.csv')
# print(files_path98
file = files_path[144]
f = mpimg.imread("./data/training/"+file)
print(f.shape)
saved = f.shape
print(saved)
img = cv2.resize(f, (224,224))
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
plt.imshow(img)
image = copy.copy(np.expand_dims(img, axis=0))
image = cv2.normalize(image, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
image/=256
key_points = model.predict_on_batch(image).reshape(-1, 2)
plt.scatter(key_points[:, 0], key_points[:, 1], marker='.', c='m')
plt.waitforbuttonpress()