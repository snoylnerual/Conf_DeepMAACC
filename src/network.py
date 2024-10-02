from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, BatchNormalization, Resizing, MaxPooling2D, Dropout, Input, Activation, ZeroPadding2D, LSTM, Embedding
from keras.datasets import fashion_mnist, mnist, cifar10, cifar100, imdb, reuters
from keras.utils import to_categorical, pad_sequences
from keras.applications import VGG16, ResNet50
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping
from keras import Sequential, layers
from argparse import ArgumentParser
from keras.regularizers import l2
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os


class Network:
    def __init__(self, model_type, Dataset):
        if model_type in dir(Network):  # and ('scratch' in model_type or 'keras' in model_type):
            self._model_type = model_type
        else:
            raise ValueError('Model type not recognized')
        self._dataset_name = Dataset.get_dataset_name()
        self._model_file_name = model_type + '-' + self._dataset_name + '.keras'
        self._x_train = Dataset.get_x_train()
        self._y_train = Dataset.get_y_train()
        self._x_test = Dataset.get_x_test()
        self._y_test = Dataset.get_y_test()
        self._nb_classes = Dataset.get_nb_classes()
        self._model = None

    def train(self):
        self._model = getattr(Network, self._model_type)(self, self._x_train, self._y_train, self._x_test, self._y_test, self._nb_classes)

    def retrieve_model(self):
        self._model = load_model('../inputs/' + self._dataset_name + '/' + self._model_file_name)

    def get_model(self):
        return self._model

    def get_nb_class(self):
        return self._nb_classes

    def fcnn(self, x_train, y_train, x_test, y_test, nb_classes):
        #nb_classes = 10
        model = Sequential()

        model.add(Flatten(input_shape=(32, 32, 3)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=0)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))  # , callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/fcnn-' + self._dataset_name + '.keras')
        print("Model saved")
        # TODO change print to logs
        return model

    def lenet5(self, x_train, y_train, x_test, y_test, nb_classes):
        #nb_classes = 10
        model = Sequential()

        model.add(Conv2D(6, kernel_size=(5, 5), activation='sigmoid', input_shape=(32, 32, 3)))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation='sigmoid'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='sigmoid'))
        model.add(Dense(84, activation='sigmoid'))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))  # , callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/lenet5-' + self._dataset_name + '.keras')
        print("Model saved")
        # TODO change print to logs
        return model

    def resnet18(self, x_train, y_train, x_test, y_test, nb_classes):
        #https://github.com/jerett/Keras-CIFAR10/blob/master/classifiers/ResNet.py
        #nb_classes = 10
        weight_decay = 1e-4
        input = Input(shape=(32, 32, 3))
        x = input
        x = Resizing(224,224)(x)
        x = self.conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
        # # conv 2
        x = self.ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        x = self.ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 3
        x = self.ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = self.ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 4
        x = self.ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = self.ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 5
        x = self.ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = self.ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)
        model = Model(input, x, name='ResNet18')

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))  # , callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/resnet18-' + self._dataset_name + '.keras')
        print("Model saved")
        # TODO change print to logs
        return model

    def rnn(self, x_train, y_train, x_test, y_test, nb_classes):
        max_features = 20000
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
        model.save('../inputs/' + self._dataset_name + '/rnn-' + self._dataset_name + '.keras')
        print("Model saved")

        return model

    def conv2d_bn(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       use_bias=True,
                       kernel_regularizer=l2(weight_decay)
                       )(x)
        layer = BatchNormalization()(layer)
        return layer

    def conv2d_bn_relu(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = self.conv2d_bn(x, filters, kernel_size, weight_decay, strides)
        layer = Activation('relu')(layer)
        return layer

    def ResidualBlock(self, x, filters, kernel_size, weight_decay, downsample=True):
        if downsample:
            # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
            residual_x = self.conv2d_bn(x, filters, kernel_size=1, strides=2)
            stride = 2
        else:
            residual_x = x
            stride = 1
        residual = self.conv2d_bn_relu(x,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  weight_decay=weight_decay,
                                  strides=stride,
                                  )
        residual = self.conv2d_bn(residual,
                             filters=filters,
                             kernel_size=kernel_size,
                             weight_decay=weight_decay,
                             strides=1,
                             )
        out = layers.add([residual_x, residual])
        out = Activation('relu')(out)
        return out

    def resnet50_keras(self, x_train, y_train, x_test, y_test):
        nb_classes = 10
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/resnet50_keras-' + self._dataset_name + '.keras')
        print("Model saved")
        return model

    def alexnet_scratch(self, x_train, y_train, x_test, y_test, nb_classes):
        # from https://medium.datadriveninvestor.com/alexnet-implementation-using-keras-7c10d1bb6715
        # Instantiate an empty model
        #nb_classes = 10
        model = Sequential()
        model.add(Input(shape=(32, 32, 3)))
        model.add(ZeroPadding2D((5, 5)))
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(10))
        model.add(Activation('softmax'))


        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/alexnet_scratch-' + self._dataset_name + '.keras')
        print("Model saved")
        return model

    def vggnet16_scratch(self, x_train, y_train, x_test, y_test, nb_classes):
        #nb_classes = 10

        model = Sequential()

        model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=nb_classes, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/vggnet16_scratch-' + self._dataset_name + '.keras')
        print("Model saved")
        return model

    def vggnet16_keras(self, x_train, y_train, x_test, y_test, nb_classes):
        #nb_classes = 10
        model = VGG16(classes=10)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('../inputs/' + self._dataset_name + '/vggnet16_keras-' + self._dataset_name + '.keras')
        print("Model saved")
        return model

class Dataset:
# mnist, fmnist, kmnist, emnist, cifar10, cifar100

    def __init__(self, dataset_name):
        # mnist/data/mnist_train_inputs.npy
        # mnist/data/mnist_train_outputs.npy
        # mnist/data/mnist_test_inputs.npy
        # mnist/data/mnist_test_outputs.npy
        self._dataset_name = dataset_name
        if self.check_file(dataset_name):
            print("Files exist")
            # TODO change print to logs
            self._x_train = np.load('../inputs/' + dataset_name + '/data/' + dataset_name + '_train_inputs.npy')
            self._y_train = np.load('../inputs/' + dataset_name + '/data/' + dataset_name + '_train_outputs.npy')
            self._x_test = np.load('../inputs/' + dataset_name + '/data/' + dataset_name + '_test_inputs.npy')
            self._y_test = np.load('../inputs/' + dataset_name + '/data/' + dataset_name + '_test_outputs.npy')
            self._nb_classes = self.get_nb_classes()
        else:
            print("Files don't exist")
            # TODO change print to logs
            self._x_train, self._y_train, self._x_test, self._y_test, self._nb_classes = self.dataset(dataset_name)

    def get_dataset_name(self):
        return self._dataset_name
    def get_x_train(self):
        return self._x_train
    def get_y_train(self):
        return self._y_train
    def get_x_test(self):
        return self._x_test
    def get_y_test(self):
        return self._y_test

    def get_nb_classes(self):
        if self._dataset_name in ['mnist', 'fmnist', 'kmnist', 'cifar10']:
            nb_classes = 10
        elif self._dataset_name in ['svhn']:
            nb_classes = 11
        elif self._dataset_name in ['emnist']:
            nb_classes = 27
        elif self._dataset_name in ['cifar100']:
            nb_classes = 100
        elif self._dataset_name in ['imdb']:
            nb_classes = 2
        elif self._dataset_name in ['reuters']:
            nb_classes = 90
        else:
            raise ValueError("Dataset not exist")
        return nb_classes

    def dataset(self,dataset_name):
        if dataset_name == 'mnist':
            print("Mnist data retrieval")
            # TODO change print to logs
            nb_classes = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)
        elif dataset_name == 'fmnist':
            print("Fmnist data retrieval")
            # TODO change print to logs
            nb_classes = 10
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)
        elif dataset_name == 'kmnist':
            nb_classes = 10
            x_train = np.load('../inputs/kmnist/data/kmnist_train_../inputs.npz')['arr_0']
            y_train = np.load('../inputs/kmnist/data/kmnist_train_outputs.npz')['arr_0']
            x_test = np.load('../inputs/kmnist/data/kmnist_test_../inputs.npz')['arr_0']
            y_test = np.load('../inputs/kmnist/data/kmnist_test_outputs.npz')['arr_0']
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)
        elif dataset_name == 'emnist':
            nb_classes = 27
            data = sio.loadmat('../inputs/emnist/data/emnist-letters')['dataset']
            x_train = data['train'][0, 0]['images'][0, 0]
            y_train = data['train'][0, 0]['labels'][0, 0]
            x_test = data['test'][0, 0]['images'][0, 0]
            y_test = data['test'][0, 0]['labels'][0, 0]
            x_train = x_train.reshape(124800,28,28)
            x_test = x_test.reshape(20800,28,28)
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train), to_categorical(y_test)

        elif dataset_name == 'cifar10':
            print("Cifar10 data retrieval")
            nb_classes = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)

        elif dataset_name == 'cifar100':
            print("Cifar100 data retrieval")
            nb_classes = 100
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)

        elif dataset_name == 'svhn':
            nb_classes = 11
            train_data = sio.loadmat('../inputs/svhn/data/train_32x32.mat')
            extrain_data = sio.loadmat('../inputs/svhn/data/extra_32x32.mat')
            test_data = sio.loadmat('../inputs/svhn/data/test_32x32.mat')
            x_train = train_data['X']
            y_train = train_data['y']
            x_extrain = extrain_data['X']
            y_extrain = extrain_data['y']
            x_train = np.concatenate((x_train, x_extrain), axis=None)
            y_train = np.concatenate((y_train, y_extrain), axis=None)
            x_test = test_data['X']
            y_test = test_data['y']
            x_train = x_train.reshape(604388, 32, 32, 3)
            x_test = x_test.reshape(26032, 32, 32, 3)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = to_categorical(y_train, num_classes=nb_classes), to_categorical(y_test, num_classes=nb_classes)

        elif dataset_name == 'imdb':
            nb_classes = 2
            max_features = 20000
            maxlen = 80
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)
            x_train, x_test = pad_sequences(x_train, maxlen=maxlen), pad_sequences(x_test, maxlen=maxlen)

        elif dataset_name == 'reuters':
            nb_classes = 90
            max_features = 20000
            maxlen = 80
            (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
            y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)
            x_train, x_test = pad_sequences(x_train, maxlen=maxlen), pad_sequences(x_test, maxlen=maxlen)

        else:
            x_train = y_train = x_test = y_test = np.array([])
            nb_classes = 10

        np.save('../inputs/' + dataset_name + '/data/' + dataset_name + '_train_../inputs', x_train)
        np.save('../inputs/' + dataset_name + '/data/' + dataset_name + '_train_outputs', y_train)
        np.save('../inputs/' + dataset_name + '/data/' + dataset_name + '_test_../inputs', x_test)
        np.save('../inputs/' + dataset_name + '/data/' + dataset_name + '_test_outputs', y_test)
        return x_train, y_train, x_test, y_test, nb_classes
        pass

    def check_file(self, file_name):
        tx = '../inputs/' + file_name + '/data/' + file_name + '_train_inputs.npy'
        ty = '../inputs/' + file_name + '/data/' + file_name + '_train_outputs.npy'
        ttx = '../inputs/' + file_name + '/data/' + file_name + '_test_inputs.npy'
        tty = '../inputs/' + file_name + '/data/' + file_name + '_test_outputs.npy'
        return os.path.isfile(tx) and os.path.isfile(ty) and os.path.isfile(ttx) and os.path.isfile(tty)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        dest='model_type',
                        help='Potential models: \'fcnn\', \'lenet5\', \'resnet18\', \'alexnet\', \'vggnet16\'',
                        required=True)
    parser.add_argument('-d',
                        '--dataset',
                        dest='dataset',
                        help='Potential values: \'mnist\', \'fmnist\', \'kmnist\', \'emnist\', \'cifar10\', \'cifar100\'',
                        required=True)
    args = parser.parse_args()

    data = Dataset(args.dataset)
    network = Network(args.model_type, data)
    network.train()



