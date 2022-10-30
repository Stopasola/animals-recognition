import sys
from pictures import Pictures
from neural_net import ConvolutionalNeuralNet


def main(database_path):
    #classes = ['dogs', 'panda', 'cats']
    pic = Pictures(database_path)
    neuralnet = ConvolutionalNeuralNet()

    pic.display_pictures(pic.classes)
    pic.pictures_shape(pic.classes)
    image_paths = pic.shuffle_image_paths(pic.classes)
    data, labels = pic.pre_process_images(image_paths)
    pic.show_labeled_image(pic.classes, labels, data)
    train_X, test_X, train_Y, test_Y = neuralnet.split_dataset(labels, data)
    classifier = neuralnet.architecture(pic.HEIGHT, pic.WIDTH, pic.N_CHANNELS)
    classifier = neuralnet.neural_network(classifier)
    classifier.fit(train_X, train_Y, batch_size=32, epochs=25, verbose=1)
    neuralnet.evaluate_classifier(classifier, test_X, test_Y, pic.classes)


"""==============================================================="""
"""==============================================================="""

N_CHANNELS = 3
HEIGHT = 32
WIDTH = 55

if __name__ == '__main__':

    database_path = sys.argv[1]
    main(database_path)
