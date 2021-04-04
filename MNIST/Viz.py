# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:15:46 2021

@author: Angad
"""

import pandas
from keras.utils import to_categorical
from matplotlib import pyplot

train_data = pandas.read_csv('./data/train.csv')
train_y = to_categorical(train_data["label"])
train_x = train_data.loc[:, train_data.columns != "label"]
train_x /= 256
train_x = train_x.values.reshape(-1, 28, 28, 1)

n_samples = 25
for i in range(n_samples):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot single image
	pyplot.imshow(train_x[i], cmap='gray_r')