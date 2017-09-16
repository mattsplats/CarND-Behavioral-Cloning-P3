import os
import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# global parameters
TEST_FOLDER = '/data'
EPOCHS = 3
CORRECTION_FACTOR = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            train_images = []
            train_angles = []
            for batch_sample in batch_samples:
                names = ['.{}/IMG/{}'.format(TEST_FOLDER, batch_sample[i].split('\\')[-1]) for i in range(3)]
                images = [cv2.imread(name) for name in names]
                angles = float(batch_sample[3]), float(batch_sample[3]) + CORRECTION_FACTOR, float(batch_sample[3]) - CORRECTION_FACTOR
                train_images.extend(images)
                train_angles.extend(angles)
                train_images.extend([cv2.flip(img, 1) for img in images])
                train_angles.extend([ang * -1.0 for ang in angles])

            # trim image to only see section with road
            X_train = np.array(train_images)
            y_train = np.array(train_angles)
            yield shuffle(X_train, y_train)


# import csv data
samples = []
with open('.{}/driving_log.csv'.format(TEST_FOLDER)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# build Keras model
model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))  # Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=EPOCHS)
print('Training complete!')

model.save('model.h5')
print('Model saved.')