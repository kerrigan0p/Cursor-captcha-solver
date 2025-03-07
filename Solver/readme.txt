Overview

This repository contains an implementation of DrawLocatorNet, a deep learning model designed to locate specific drawn elements within a larger field image. The model uses two convolutional neural network (CNN) branches to extract features from both the field image and the smaller drawn elements, and then predicts the coordinates of the drawn elements within the field.

Requirements

To run this project, you need the following Python libraries:
cv2 (OpenCV): For image loading and processing.
numpy: For numerical computations.
torch, torch.nn, torch.nn.functional: For defining and training the neural network.
torchvision.transforms: For preprocessing images.
PIL (Pillow): For image conversions.
pandas: (not directly used in the provided script but imported—can be removed if unnecessary).

You can install the required libraries using:

pip install opencv-python numpy torch torchvision pillow pandas

Files in the Repository

draw_locator.py: The main script containing the model definition, utility functions, and inference pipeline.
model.pth: (Expected) Trained model file used for inference. You need to provide a trained model checkpoint.
images/: Directory containing sample images for testing.

Model Architecture

The DrawLocatorNet consists of two CNN branches:

Field CNN: Processes a 210x210 field image and extracts features using convolutional layers.

Draw CNN: Processes a 50x50 drawn element image and extracts features.

Correlation Computation: Computes a similarity map between extracted features and applies a soft-argmax to predict (x, y) coordinates of the drawn element within the field image.

Functions Explanation

DrawLocatorNet(nn.Module) :
Defines the CNN architecture.
Processes two inputs: field_img (210x210) and draw_img (50x50).
Computes the correlation map to predict the drawn element's location.
Outputs normalized coordinates (x, y) in the range [0,1].

get_information_zone(img) :
Crops the input image to extract three key regions:
field_img: The main field where the elements are located.
draw1_img and draw2_img: Two small regions containing the drawn elements.

cv2_to_pil(cv2_img) :

Converts an OpenCV image (BGR format) to a PIL image (RGB format) for compatibility with PyTorch.

make_predictions(field_img, draw1_img, draw2_img, model_path, device) :

Loads the trained model from model_path.
Preprocesses the input images and passes them through the model.
Converts the normalized predictions into pixel coordinates relative to the original image.
run(img, model_path="model.pth")

The main function that:

Extracts the relevant regions from the input image.
Converts them to the correct format.
Runs inference using the trained model.
Returns the predicted coordinates of the drawn elements.