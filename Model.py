
# coding: utf-8

# In[1]:

import zipfile
with zipfile.ZipFile("data.zip","r") as zip_ref:
    zip_ref.extractall("./")


# In[3]:

import csv
import cv2
import numpy as np
import sklearn
import os


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
correction = 0.15

# from sklearn.model_selection import train_test_split
# train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# def generator(samples, batch_size=32):
#     num_samples = len(lines)
#     while 1: # Loop forever so the generator never terminates
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]

#             images = []
#             angles = []
#             for batch_sample in batch_samples:
#                 name = './IMG/'+batch_sample[0].split('\\')[-1]
#                 center_image = cv2.imread(name)
#                 center_angle = float(batch_sample[3])
#                 images.append(center_image)
#                 angles.append(center_angle)
                
#             augmented_images, augmented_angles = [],[]
#             for image,angle in zip(images, angles):
#                 augmented_images.append(image)
#                 augmented_angles.append(angle)
#                 augmented_images.append(cv2.flip(image,1))
#                 augmented_angles.append(angle*-1)

#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             yield sklearn.utils.shuffle(X_train, y_train)



for line in lines[1:-1]:
    source_path_center = line[0]
    filename_center =source_path_center.split('\\')[-1]
    current_path_center = './data/IMG/' + filename_center
    image_center = cv2.imread(current_path_center)
    images.append(image_center)
    
#     source_path_left = line[1]
#     filename_left =source_path_left.split('\\')[-1] 
#     current_path_left = './data/IMG/' + filename_left
#     image_left = cv2.imread(current_path_left)  
#     images.append(image_left)
#     source_path_right = line[2]
#     filename_right =source_path_right.split('\\')[-1] 
#     current_path_right = './data/IMG/' + filename_right
#     image_right = cv2.imread(current_path_right)
#     images.append(image_right)
       
    measurement_center = float(line[3])
    measurement_left = float(line[3]) + correction
    measurement_right = float(line[3]) - correction
    measurements.append(measurement_center)
#     measurements.append(measurement_left)
#     measurements.append(measurement_right)

augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)
    
# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

print('images[0].shape:', images[0].shape)


model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3), output_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Flatten())
model.add(Dense(1))


model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=3)

# # model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')


# In[4]:

print('images[0].shape:', images[0].shape)


# In[ ]:



