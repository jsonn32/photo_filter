import os
import cv2
import numpy as np

def classify_and_filter_images(images, filenames, model, threshold=0.5):
    predictions = model.predict(images)
    house_indices = [i for i, p in enumerate(predictions) if p > threshold]
    non_house_indices = [i for i, p in enumerate(predictions) if p <= threshold]

    house_images_filenames = [filenames[i] for i in house_indices]
    non_house_images_filenames = [filenames[i] for i in non_house_indices]

    return house_images_filenames, non_house_images_filenames

def save_filtered_images(filenames, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in filenames:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_folder, filename), img)
