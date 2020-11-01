# -*- coding: utf-8 -*-
"""Glove-Detection.ipynb

"""

# from google.colab import drive
# drive.mount('/content/drive')

import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from os import listdir
import cv2
from pathlib import Path
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

''' Preparing the data'''

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData
        if(os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file i.e zero size')
    
    train_set_length = int(len(dataset) * SPLIT_SIZE)
    test_set_length = int(len(dataset) - train_set_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_set_length]
    test_set = dataset[-test_set_length:]
       
    for unitData in train_set:
        temp_train_set = SOURCE + unitData
        final_train_set = TRAINING + unitData
        copyfile(temp_train_set, final_train_set)
    
    for unitData in test_set:
        temp_test_set = SOURCE + unitData
        final_test_set = TESTING + unitData
        copyfile(temp_test_set, final_test_set)
        
        
YES_SOURCE_DIR = "gloves-dataset/trial1/augmented data1/yesreal/"
TRAINING_YES_DIR = "gloves-dataset/trial1/augmented data1/training/yes1/"
TESTING_YES_DIR = "gloves-dataset/trial1/augmented data1/testing/yes1/"
NO_SOURCE_DIR = "gloves-dataset/trial1/augmented data1/noreal/"
TRAINING_NO_DIR = "gloves-dataset/trial1/augmented data1/training/no1/"
TESTING_NO_DIR = "gloves-dataset/trial1/augmented data1/testing/no1/"
split_size = .8
split_data(YES_SOURCE_DIR, TRAINING_YES_DIR, TESTING_YES_DIR, split_size)
split_data(NO_SOURCE_DIR, TRAINING_NO_DIR, TESTING_NO_DIR, split_size)

'''Training the model for Glove Detection'''

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2,2),
    
#     tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
    
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(50, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# TRAINING_DIR = "gloves-dataset/trial1/augmented data1/training"
# train_datagen = ImageDataGenerator(rescale=1.0/255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')

# train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
#                                                     batch_size=10, 
#                                                     target_size=(150, 150))
# VALIDATION_DIR = "gloves-dataset/trial1/augmented data1/testing"
# validation_datagen = ImageDataGenerator(rescale=1.0/255)

# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
#                                                          batch_size=10, 
#                                                          target_size=(150, 150))
# checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

# history = model.fit_generator(train_generator,
#                               epochs=30,
#                               validation_data=validation_generator,
#                               callbacks=[checkpoint])

# model.save('trained_model.h5')


'''Loding the pre-trained model'''

model = load_model('trained_model.h5')
face_clsfr=cv2.CascadeClassifier('palm.xml')

labels_dict={0:'No_Glove',1:'Glove'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) 

'''Making use of a harrcascade file for detecting the hand'''
classifier = cv2.CascadeClassifier('palm.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flipping the image

    # Resizing image for faster detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale
    hands = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in hands:
        (x, y, w, h) = [v * size for v in f] #Scaleing the shapesize backup
        #Rectangle around the hands
        hand_img = im[y:y+h, x:x+w]
        resized=cv2.resize(hand_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Displaying the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # Exit via the escape key 
    if key == 27: 
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()