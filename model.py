import os
import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


# global parameters
TEST_FOLDER = '/data'
EPOCHS = 3

IMG_PER_SAMPLE = 6  # each steering angle has 6 associated images, 2 per camera (augmented with horizontal flip)
VAL_PCT = 0.3       # 70% training / 30% validation split

CORRECTION_FACTOR = 0.2  # steering angle correction factor for left/right camera images
KEEP_PROB_CONV = 0.25    # dropout %'s for convolutional and fully connected layers
KEEP_PROB_FC = 0.5


# Python generator for loading sample batches
def generator(samples, batch_size=32):
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            train_images = []
            train_angles = []
            for batch_sample in batch_samples:
                # read images and steering angles
                names = ['.{}/IMG/{}'.format(TEST_FOLDER, batch_sample[i].split('\\')[-1]) for i in range(3)]
                images = [cv2.imread(name) for name in names]
                angles = float(batch_sample[3]), float(batch_sample[3]) + CORRECTION_FACTOR, float(batch_sample[3]) - CORRECTION_FACTOR

                # extend data with new images/angles
                train_images.extend(images)
                train_angles.extend(angles)

                # extend data with augmented images/angles (horizonal flip)
                train_images.extend([cv2.flip(img, 1) for img in images])
                train_angles.extend([ang * -1.0 for ang in angles])

            X_train = np.array(train_images)
            y_train = np.array(train_angles)
            yield shuffle(X_train, y_train)


# import csv data
samples = []
with open('.{}/driving_log.csv'.format(TEST_FOLDER)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# split samples into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=VAL_PCT)

# setup generators for training and validation
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# build Keras model
model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))  # Preprocess incoming data, centered around zero
model.add(Cropping2D(cropping=((70,25), (0,0))))                      # Crop upper and lower parts of image (sky and car/wheels) 

# convolutional layers
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(KEEP_PROB_CONV))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(KEEP_PROB_CONV))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(KEEP_PROB_CONV))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(KEEP_PROB_FC))
model.add(Dense(50))
model.add(Dropout(KEEP_PROB_FC))
model.add(Dense(10))

# single output for regression network
model.add(Dense(1))

# train the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * IMG_PER_SAMPLE, validation_data=validation_generator, nb_val_samples=len(validation_samples) * IMG_PER_SAMPLE, nb_epoch=EPOCHS)
print('Training complete!')

# save the model
model.save('model.h5')
print('Model saved.')