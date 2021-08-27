import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

def main():
    (X_training, y_training), (X_test, y_test) = cifar100.load_data()
    plt.imshow(X_training[1], cmap = 'gray') # Show image, cmap = gray remove the color of the image
    plt.title('Class ' + str(y_training[1])) # Class of the image

    predictors_training = X_training.reshape(X_training.shape[0], 32, 32, 1)

    predictors_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
    predictors_training = predictors_training.astype('float32')
    predictors_test = predictors_test.astype('float32')

    predictors_training /= 255
    predictors_test /= 255

    class_training = np_utils.to_categorical(y_treinamento, 100)
    class_test = np_utils.to_categorical(y_test, 100)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
