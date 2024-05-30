import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import UnidentifiedImageError

def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    for filename in os.listdir(folder):
        if not filename.lower().endswith(valid_extensions):
            print(f"Skipping non-image file {filename}")
            continue
        img_path = os.path.join(folder, filename)
        try:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            if label is not None:
                labels.append(label)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    return images, labels

def load_and_preprocess_images(folder, batch_size=100):
    images = []
    filenames = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    file_list = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i + batch_size]
        batch_images = []
        batch_filenames = []
        for filename in batch_files:
            img_path = os.path.join(folder, filename)
            try:
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
                batch_filenames.append(filename)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        images.extend(batch_images)
        filenames.extend(batch_filenames)
        print(f"Loaded batch {i // batch_size + 1}")
    
    return np.array(images), filenames
