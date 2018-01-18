'''
Udacity - Project-3 Behavior Cloning
Author: Vidit Shah
Contact:vidit02100@gmail.com
'''
#Importing all the Libraries

import os
import csv
import cv2
import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Cropping2D, Lambda, Conv2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#Defining Hyper Parameters and path ways for the Program

BATCH_SIZE = 128
EPOCHS = 10
MAIN_PATH = '/Users/ViditShah/Desktop/Vdesktop/SELFDRIVINGCARNANODEGREE/'
DATA_FILE = 'driving_log.csv'
IMGS_PATH = '/Users/ViditShah/Desktop/Vdesktop/SELFDRIVINGCARNANODEGREE/IMG'
LEARNING_RATE = 1e-4

#Fuction to load our Data from Where we Have recorded it.
def load_dataset():
    data_file = os.path.join(MAIN_PATH, DATA_FILE)
    X_train, y_train = [], []

    with open(data_file, newline='') as log_file:
        log_file_reader = csv.reader(log_file)

        for row in log_file_reader:
            if (row[0] == 'center'):
                next(log_file_reader)
            else:
                X_train.append([row[0], row[1], row[2]])  #Feature
                y_train.append(row[3])  #Label

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        return X_train, X_val, y_train, y_val

#With help of Opencv Reading Each Image

def load_image(img):
    img_path = os.path.join(IMGS_PATH, img.strip())
    img_read = cv2.imread(img_path)
    return cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)


def generator(X, y):

    correction_factor = 0.2

    X_aug = []
    y_aug = []

    #LIST FROM LEFT TO RIGHT AND APPLY SOME BASIC AUGMENTATION
    for img, angle in zip(X, y):
        X_aug.extend([img[0], img[1], img[2]])
        y_aug.extend([float(angle), float(angle) + correction_factor, float(angle) - correction_factor])

    X_aug, y_aug = shuffle(X_aug, y_aug)

    n_samples = len(X_aug)
    print("\nNumber of samples = ", n_samples)

    while 1:  # Loop forever so the generator never terminates

        for offset in range(0, n_samples, BATCH_SIZE):

            images_batch = X_aug[offset:offset + BATCH_SIZE]
            angle_batch = y_aug[offset:offset + BATCH_SIZE]

            images = []
            angles = []

            for im, angle in zip(images_batch, angle_batch):
                image = load_image(im)
                images.append(image)
                angles.append(float(angle))

            X_batch = np.array(images)
            y_batch = np.array(angles)

            yield shuffle(X_batch, y_batch)

#Defining NVIDIA Model.
def nvidia_model(resize_factor=0.5):

    input_shape = (160, 320, 3)
    i = Input(shape=input_shape)

    x = Lambda(lambda x: x / 127.5 - 1.)(i)
    x = Cropping2D(cropping=((50,20), (0,0)))(x)


    x = Conv2D(24, (5, 5), activation='relu', strides=(2,2))(x)
    x = Conv2D(36, (5, 5), activation='relu', strides=(2,2))(x)
    x = Conv2D(48, (5, 5), activation='relu', strides=(2,2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1)(x)

    m = Model(i, x)
    m.summary()

    return m

def plot_loss(obj):
    print(obj.history.keys())
    plt.plot(obj.history['loss'])
    plt.plot(obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':

    # Load the file names for the train and validation features
    X_train, X_val, y_train, y_val = load_dataset()

    # Model definition
    model = nvidia_model()

    # model compilation
    model.compile(loss='mse', optimizer=Adam(LEARNING_RATE))

    # define the train an validation generators
    train_generator = generator(X_train, y_train)
    validation_generator = generator(X_val, y_val)

    t0 = time.time()

    steps_per_epoch = 3 * len(X_train) / BATCH_SIZE  # The data is augmented with the 3 cameras
    validation_steps = len(X_val) / BATCH_SIZE  # The validation data is keep to the original size

    # Train the model
    train_obj = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=2,
                                     validation_data=validation_generator, validation_steps=validation_steps)

    print("Time: %.3f seconds" % (time.time() - t0))

    # Save model
    model.save('model.h5')
    # plot results
    plot_loss(train_obj)

    from keras import backend as K
    del model
    K.clear_session()
