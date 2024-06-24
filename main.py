import torch
import torch.nn as nn
import shutil
import os
from torchvision import datasets, transforms, models
from torchvision import transforms
from PIL import Image

AlexNetmodel = models.alexnet(pretrained = True)  # Loading PreTrained Network
AlexNetmodel


AlexNetmodel.classifier = nn.Sequential(nn.Linear(9216,1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.4),
                                        nn.Linear(1024,2),
                                        nn.LogSoftmax(dim=1))

AlexNetmodel.load_state_dict(torch.load('alexnet_model.pth'))
AlexNetmodel.eval()  # Setting Model to evaluation mode

# Creating two folders to save good and bad images in there distinct folders
os.makedirs('good_images', exist_ok=True)
os.makedirs('bad_images', exist_ok=True)


# Define the transformation for the test images
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the Testing folder
test_folder_path = 'Images'  # Change this folder accoring to your path 

# List all images in the Testing folder
image_paths = [os.path.join(test_folder_path, img) for img in os.listdir(test_folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]

# Loop through each image and classify it
for image_path in image_paths:
    # Open the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    image_tensor = test_transforms(image).unsqueeze(0)  # Add batch dimension
    
    # Run the model on the image
    with torch.no_grad():
        output = AlexNetmodel(image_tensor)
        _, predicted = torch.max(output, 1)
        
    # Get the predicted class (assuming 0 for bad and 1 for good)
    class_name = 'good' if predicted.item() == 1 else 'bad'
    
    # Define the target folder
    target_folder = 'good_images' if class_name == 'good' else 'bad_images'
    
    # Save the image in the respective folder
    shutil.copy(image_path, os.path.join(target_folder, os.path.basename(image_path)))

print('Classification and saving completed.')
