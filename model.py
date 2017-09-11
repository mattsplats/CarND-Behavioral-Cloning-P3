import os
import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda


test_folder = '/straight_laps'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '.{}/IMG/{}'.format(test_folder, batch_sample[0].split('\\')[-1])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# import csv data
samples = []
with open('.{}/driving_log.csv'.format(test_folder)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# build Keras model
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation
shape = 160, 320, 3  # Trimmed image format
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=shape))

model.add(Flatten(input_shape=shape))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')