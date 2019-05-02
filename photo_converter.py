from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection
import tensorflow as tf

path = ""
classes = ["redcrestedcardinal", "sparrow", "chicken"]
num_classes = len(classes)
image_size = 50

# Reading picture files

X = []
Y = []
for index, aclass in enumerate(classes):
    photos_dir = path + aclass
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i > 300:
            break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.asarray(X)
Y = np.asarray(Y)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)
np.save("./birds.npy", xy)

