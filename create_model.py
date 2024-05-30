from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Define directories
saved_folder = 'saved'
deleted_folder = 'deleted'

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create generators
train_generator = datagen.flow_from_directory(
    directory='.',
    classes=['saved', 'deleted'],
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Create and compile the model
model = create_model()

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10
)

# Save the model for future use
model.save('house_classifier_model.h5')
