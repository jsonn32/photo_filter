import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_images_from_folder, load_and_preprocess_images
from model import create_model
from filter_images.py import classify_and_filter_images, save_filtered_images

# Load and preprocess data
house_folder = 'house_images'
non_house_folder = 'non_house_images'
house_images, house_labels = load_images_from_folder(house_folder, label=1)
non_house_images, non_house_labels = load_images_from_folder(non_house_folder, label=0)

X = np.array(house_images + non_house_images)
y = np.array(house_labels + non_house_labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = create_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Filter new images
input_folder = 'path_to_folder_with_new_images'
output_folder = 'path_to_filtered_images_folder'
new_images, filenames = load_and_preprocess_images(input_folder)
house_images_filenames, non_house_images_filenames = classify_and_filter_images(new_images, filenames, model)

# Save filtered images
save_filtered_images(house_images_filenames, input_folder, output_folder)

print("House images:", house_images_filenames)
print("Non-house images:", non_house_images_filenames)
