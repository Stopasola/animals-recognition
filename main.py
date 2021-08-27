import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

def main():
    # loading dataset
    (X_training, y_training), (X_test, y_test) = cifar100.load_data()
    plt.imshow(X_training[1], cmap = 'gray') # Show image, cmap = gray remove the color of the image
    plt.title('Class ' + str(y_training[1])) # Class of the image

    predictors_training = X_training.reshape(3, 32, 32, 1)

    predictors_test = X_test.reshape(3, 32, 32, 1)
    predictors_training = predictors_training.astype('float32')
    predictors_test = predictors_test.astype('float32')

    predictors_training /= 255
    predictors_test /= 255

    class_training = np_utils.to_categorical(y_treinamento, 100)
    class_test = np_utils.to_categorical(y_test, 100)

    #plt.show()

    # Creating classifier
    classifier = Sequential()
    classifier.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation = 'relu')) #recommended start with 64 kernels instead of 32

    #Pooling
    classifier.add(MaxPooling2D(pool_size = (2,2)))

    #Flattening
    classifier.add(Flatten())

    #Dense Neural Net
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 10, activation = 'softmax')) #output layer
    classifier.compile(loss = 'categorical_crossentropy',
                       optimizer = 'adam', metrics = ['accuracy'])
    classifier.fit(predictors_training, class_training,
                   batch_size = 128, epochs = 5,
                   validation_data = (predictors_test, class_test))

    result = classifier.evaluate(predictors_test, class_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
