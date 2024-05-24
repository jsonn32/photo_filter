import os
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(224, 224))
        if img is not None:
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            if label is not None:
                labels.append(label)
    return images, labels

def load_and_preprocess_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(224, 224))
        if img is not None:
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            filenames.append(filename)
    return np.array(images), filenames
