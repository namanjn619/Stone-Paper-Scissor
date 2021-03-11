from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import os
import cv2
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

image_path = "C://Users//a//Desktop//VScode//Rock Paper Scissor//My Game//Images//"
class_map = {
    "Rock": 0,
    "Paper": 1,
    "Scissor": 2,
}
num_class = len(class_map)

def mapper(val):
    return class_map[val]

model = Sequential()
model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        strides=(1,1),
                        activation='relu',
                        input_shape=(227,227,3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(filters=64,
                        kernel_size=(3,3),
                        activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(filters=64,
                        kernel_size=(3,3),
                        activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,
                activation='relu'))
model.add(Dense(units=64,
               activation='relu'))
model.add(Dense(units=32,
               activation='relu'))
model.add(Dense(units=16,
               activation='relu'))
model.add(Dense(units=8,
               activation='relu'))
model.add(Dense(units=4,
               activation='relu'))
model.add(Dense(units=3,
               activation='softmax'))


dataset = []
for directory in os.listdir(image_path):
    path = os.path.join(image_path,directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

data, labels = zip(*dataset)
labels = list(map(mapper, labels))

labels = np_utils.to_categorical(labels)

model.compile(optimizer=Adam(lr=0.0001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(np.array(data), np.array(labels), epochs=25)

model.save("rock-paper-scissors-model.h5")
print("Model Saved!!")