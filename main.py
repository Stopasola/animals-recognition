import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import random
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    # path to images
    path = 'C:/Users/andre/OneDrive/Documents/Aula/IA - Adriana/Aula8/Trabalho/animals-recognition/animals/animals/'

    # animal classes
    classes = ['dogs', 'panda', 'cats']

    #1. Display pictures
    for category in classes:
    	fig, _ = plt.subplots(3,4)
    	fig.suptitle(category)
    	for k, v in enumerate(os.listdir(path+category)[:12]):
    		img = plt.imread(path+category+'/'+v)
    		plt.subplot(3, 4, k+1)
    	plt.show()

    #2. Pictures shape
    shape0 = []
    shape1 = []

    for category in classes:
    	for files in os.listdir(path+category):
    		shape0.append(plt.imread(path+category+'/'+ files).shape[0])
    		shape1.append(plt.imread(path+category+'/'+ files).shape[1])

    	print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    	print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    	shape0 = []
    	shape1 = []

    #3. Preprocess data and label inputs

    data = []
    labels = []
    imagePaths = []
    HEIGHT = 32
    WIDTH = 55
    N_CHANNELS = 3

    for k, category in enumerate(classes):
    	for f in os.listdir(path+category):
    		imagePaths.append([path+category+'/'+f, k]) # k=0 : 'dogs', k=1 : 'panda', k=2 : 'cats'

    random.shuffle(imagePaths)
    print(imagePaths[:10])

    for imagePath in imagePaths:
    	image = cv2.imread(imagePath[0])
    	image = cv2.resize(image, (WIDTH, HEIGHT)) # .flatten()
    	data.append(image)
    	label = imagePath[1]
    	labels.append(label)


    data = np.array(data, dtype="float") / 255.0
    labels= np.array(labels)

    plt.subplots(3,4)
    for i in range(12):
    	plt.subplot(3, 4, i+1)
    	plt.imshow(data[i])
    	plt.axis('off')
    	plt.title(classes[labels[i]])
    plt.show()


    #4. Split dataset into train and test set
    (train_X, test_X, train_Y, test_Y) = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_Y = np_utils.to_categorical(train_Y, 3)

    print(train_X.shape)
    print(test_X.shape)
    print(train_Y.shape)
    print(test_Y.shape)


    #5. Define model architecture

    # Creating classifier
    classifier = Sequential()

    #Convolution layer
    classifier.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS))) #64 layers is recomended
    #classifier.add(BatchNormalization())

    #Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    #Convolution layer
    classifier.add(Convolution2D(32, (2, 2), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25)) #Avoid overfitting
    classifier.add(Flatten())

    # Neural net
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.5)) #Avoid overfitting
    classifier.add(Dense(128, activation = 'relu'))
    classifier.add(Dropout(0.5)) #Avoid overfitting
    classifier.add(Dense(3, activation='softmax'))

    classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    print(classifier.summary())

    #6. Fit classifier on training data
    classifier.fit(train_X, train_Y, batch_size=32, epochs=25, verbose=1)


    #7. Evaluate classifier on test data
    pred = classifier.predict(test_X)
    predictions = argmax(pred, axis=1) # return to label

    cm = confusion_matrix(test_Y, predictions)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Model confusion matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)

    for i in range(3):
    	for j in range(3):
    		ax.text(i, j, cm[j, i], va='center', ha='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    accuracy = accuracy_score(test_Y, predictions)
    print('Accuracy : %.2f%%' % (accuracy*100.0))


if __name__ == '__main__':
    main()
