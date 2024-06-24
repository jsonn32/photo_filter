# House Image Classifier

This repository contains a trained model that classifies images of houses into "good" and "bad" folders.

## Requirements

Before you begin, ensure you have the following installed:

- Python 3.x
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/jsonn32/photo_filter.git
    cd photo_filter
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your images in a folder called `Images` at the root of the project.

2. Run the classifier script:

    ```sh
    python main.py
    ```

3. The script will process the images and place them into `good_images` and `bad_images` folders based on the classification.

