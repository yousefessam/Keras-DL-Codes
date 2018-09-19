from __future__ import print_function
from HelperFunctionsDeepLearning2 import load_data
from HelperFunctionsDeepLearning2 import print_mislabeled_images
#Import Dataset, neural network
import keras
from keras.datasets import mnist
from keras import Sequential, optimizers
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

import os
import psutil
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from HelperFunctionsDeepLearning2 import *





plt.figure(1)
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

mybatch_size = 209
num_classes = 1
myepochs = 2500

# input image dimensions
img_rows, img_cols = 64, 64

#Load the dataset
#Split the dataset into train set and test set
x_train, y_train, x_test, y_test, classes = load_data()


# building the input vector from the 28x28 pixels
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0],-1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.T
y_test = y_test.T

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

x_train /= 255
x_test /= 255

input_shape = img_rows * img_cols * 3

# normalizing the data to help with the training

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(20, activation='relu', input_dim=input_shape))
#model.add(Dropout(0.5))
model.add(Dense(7, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))

#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))

#sgd = SGD(lr=0.0075, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.0075)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy',
 #             optimizer=sgd,
  #            metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=myepochs,
          batch_size=mybatch_size)
score = model.evaluate(x_test, y_test, batch_size=mybatch_size)



#model.predict(x_test[0].reshape(x_test[0].shape[0],1).T)

res2 = ((model.predict(x_test)>0.5).astype(int)).astype(int)
testAccuracy = len(np.where(res2 == 1)[0])/len(y_test)
print(testAccuracy)


num_images = x_test.shape[0]
plt.rcParams['figure.figsize'] = (100.0, 100.0) # set default size of plots
for i in range(0, x_test.shape[0]):
    resImage =x_test[i].reshape(img_rows,img_cols,3)
    plt.subplot(2,num_images,i+1)
    plt.imshow(resImage, interpolation='nearest')
    plt.axis('off')
    plt.title("Cat" if model.predict(x_test[i].reshape(12288,1).T) > 0.5 else "Non Cat")
