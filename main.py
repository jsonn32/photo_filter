import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_and_preprocess_images

# Load the pre-trained model
model = load_model('house_classifier_model.h5')

# Filter new images
input_folder = 'mixed_images'
output_saved_folder = 'filtered_saved'
output_deleted_folder = 'filtered_deleted'
new_images, filenames = load_and_preprocess_images(input_folder)
predictions = model.predict(new_images)
threshold = 0.5
house_indices = [i for i, p in enumerate(predictions) if p > threshold]
non_house_indices = [i for i, p in enumerate(predictions) if p <= threshold]

house_images_filenames = [filenames[i] for i in house_indices]
non_house_images_filenames = [filenames[i] for i in non_house_indices]

print("House images:", house_images_filenames)
print("Non-house images:", non_house_images_filenames)

# Save filtered images
import cv2
import os

def save_filtered_images(filenames, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in filenames:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_folder, filename), img)

save_filtered_images(house_images_filenames, input_folder, output_saved_folder)
save_filtered_images(non_house_images_filenames, input_folder, output_deleted_folder)
