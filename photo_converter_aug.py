from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

path = ""
classes = ["redcrestedcardinal", "sparrow", "chicken"]
num_classes = len(classes)
image_size = 50
num_test_data = 100
# Reading picture files

X_train = []
Y_train = []
X_test = []
Y_test = []

for index, aclass in enumerate(classes):
    photos_dir = path + aclass
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 300:
            break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)

        if i < num_test_data:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                image_r = image.rotate(angle)
                data = np.asarray(image_r)
                X_train.append(data)
                Y_train.append(index)

                image_t = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(image_t)
                X_train.append(data)
                Y_train.append(index)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./birds_aug.npy", xy)

