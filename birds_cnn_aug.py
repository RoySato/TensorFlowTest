from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.optimizers import rmsprop
import numpy as np

classes = ["redcrestedcardinal", "sparrow", "chicken"]
num_classes = len(classes)
image_size = 50

# main function
def main():
    X_train, X_test, Y_train, Y_test = np.load("./birds_aug.npy", allow_pickle=True)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)


def model_train(X_train, Y_train):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0,25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0, 25))

    model.add(Flatten())
    model.add(Dense(512))

    # relu 関数。ネガティブは捨てる
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 画像の種類数
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=100)

    # Saving Model
    model.save('./birds_cnn_aug.h5')

    return model


def model_eval(model, X_test, Y_test):
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss: ', scores[0])
    print('Test Accuracy: ', scores[1])


if __name__ == "__main__":
    main()

