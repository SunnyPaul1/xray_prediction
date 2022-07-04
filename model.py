#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
from os import listdir
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split


#setting path and creating empty list
dir = 'dataset'
root_dir = listdir(dir)
image_list, label_list = [], []

#reading and converting image to numpy array

for directory in root_dir:
    for files in listdir(f"{dir}/{directory}"):
        image_path = f"{dir}/{directory}/{files}"
        image = cv2.imread(image_path)
        image = img_to_array(image)
        image_list.append(image)
        label_list.append(directory)


#visualize the number of classes count
label_counts = pd.DataFrame(label_list).value_counts()

#storing number of classes
num_classes = len(label_counts)

#checking the label list
label_list = np.array(label_list)

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=10)

#Normalize and reshape data
x_train = np.array(x_train, dtype='object') #np.array(x_train, dtype=np.float16) / 225.0
x_train = x_train/255
x_test = np.array(x_test, dtype = 'object') #np.array(x_test, dtype=np.float16) / 225.0
x_test = x_test / 255
x_train = x_train.reshape(224, 224)
x_test = x_test.reshape(224, 224)

# Lable Binarizing
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


# splitting the training data set into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


# Building model architechture
model = Sequential()
model.add(Conv2D(8, (3,3), padding="same", input_shape=(224, 224, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))


# Compiling model
model.compile(loss = 'categorical_crossentrophy', optimizer = Adam(0.0005), matrics=['accuracy'])

# Training the model
epochs = 25
batch_size = 128
history = model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, validation_data = (x_val, y_val))

# Saving model
model.save("xray_model.h5")
