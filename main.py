import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from sklearn.model_selection import StratifiedKFold

def CrossValidation():
    seed = 5
    np.random.seed(seed)

    (X, y), (X_teste, y_teste) = cifar100.load_data()
    predictors = X.reshape(X.shape[0], 32, 32, 3)
    predictors = predictors.astype('float32')
    predictors /= 255
    classes = np_utils.to_categorical(y, 100)

    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    results = []

    a = np.zeros(5)
    b = np.zeros(shape = (classes.shape[0], 1))

    for training_index, test_index in kfold.split(predictors,
                                                  np.zeros(shape = (classes.shape[0], 1))):
        #print('Training indexes: ', training_index, 'Test indexes ', test_index)
        classifier = Sequential()
        classifier.add(Conv2D(32, (3,3), input_shape=(32, 32, 3),
                       activation = 'relu')) #recommended start with 64 kernels instead of 32 input_shape=(28, 28, 1)
        classifier.add(MaxPooling2D(pool_size = (2,2)))
        classifier.add(Flatten())
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 10, activation = 'softmax')) #hidden layer
        classifier.compile(loss = 'categorical_crossentropy',
                           optimizer = 'adam', metrics = ['accuracy'])
        classifier.fit(predictors[training_index], classes[training_index],
                       batch_size = 128, epochs = 5)
        precision = classifier.evaluate(predictors[test_index], classes[test_index])
        results.append(precision[1])

    mean = sum(results) / len(results)

def main():
    # loading dataset
    (X_training, y_training), (X_test, y_test) = cifar100.load_data()
    plt.imshow(X_training[0]) # Show image, cmap = gray remove the color of the image
    plt.title('Class ' + str(y_training[0])) # Class of the image

    predictors_training = X_training.reshape(X_training.shape[0], 32, 32, 3)

    predictors_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    predictors_training = predictors_training.astype('float32')
    predictors_test = predictors_test.astype('float32')

    predictors_training /= 255
    predictors_test /= 255

    class_training = np_utils.to_categorical(y_training, 100)
    class_test = np_utils.to_categorical(y_test, 100)

    #plt.show()

    # Creating classifier
    classifier = Sequential()
    classifier.add(Conv2D(32, (3,3), input_shape=(32, 32, 3),
                   activation = 'relu')) #recommended start with 64 kernels instead of 32 input_shape=(28, 28, 1)

    classifier.add(BatchNormalization())

    #Pooling
    classifier.add(MaxPooling2D(pool_size = (2,2)))

    #Flattening
    #classifier.add(Flatten())

    #Addition of one more convolution layer
    classifier.add(Conv2D(32, (3,3), activation = 'relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))

    #Flattening
    classifier.add(Flatten())

    #Dense Neural Net
    classifier.add(Dense(units = 128, activation = 'relu'))

    classifier.add(Dropout(0.2)) #Avoid overfitting

    classifier.add(Dense(units = 128, activation = 'relu')) #hidden layer

    classifier.add(Dropout(0.2)) #Avoid overfitting

    classifier.add(Dense(units = 100, activation = 'softmax')) #output layer
    classifier.compile(loss = 'categorical_crossentropy',
                       optimizer = 'adam', metrics = ['accuracy'])

    predictors_training = np_utils.to_categorical(predictors_training, 100)
    class_training = np_utils.to_categorical(class_training, 100)

    classifier.fit(predictors_training, class_training,
                   batch_size = 128, epochs = 5,
                   validation_data = (predictors_test, class_test))

    result = classifier.evaluate(predictors_test, class_test)
    print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
