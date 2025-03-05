import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from PIL import Image

class DrawLocatorNet(nn.Module):
    def __init__(self):
        super(DrawLocatorNet, self).__init__()

        # CNN branch for the field image (input size: 210x210)
        self.field_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Convolutional layer (B,32,210,210)
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(2),  # Downsample to (B,32,105,105)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,105,105)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,64,52,52)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B,128,52,52)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (B,128,26,26)
        )

        # CNN branch for the draw image (input size: 50x50)
        self.draw_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B,32,50,50)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,32,25,25)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,25,25)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,64,12,12)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B,128,12,12)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Compress to (B,128,1,1)
        )

    def forward(self, field_img, draw_img):
        """
        Forward pass of the model.
        
        Args:
            field_img: Tensor (B,3,210,210) - Field Image
            draw_img: Tensor (B,3,50,50) - Draw Image
            
        Returns:
            preds: Tensor (B,2) - Predicted (x, y) coordinates normalized in [0,1].
        """
        # Extract field image features (B,128,26,26)
        field_feat = self.field_cnn(field_img)

        # Extract draw image features (B,128,1,1) and flatten to (B,128)
        draw_feat = self.draw_cnn(draw_img)
        draw_feat = draw_feat.view(draw_feat.size(0), -1)

        # Compute similarity map (B,26,26) between draw image and field feature map
        correlation = (field_feat * draw_feat.view(draw_feat.size(0), draw_feat.size(1), 1, 1)).sum(dim=1)

        # Get correlation map dimensions
        B, H, W = correlation.size()

        # Generate grid for spatial coordinates
        device = correlation.device
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        # Compute probability distribution using softmax
        correlation_flat = correlation.view(B, -1)  # Flatten (B, H*W)
        prob = F.softmax(correlation_flat, dim=1).view(B, H, W)

        # Compute expected (x, y) coordinates using soft-argmax
        pred_x = (prob * grid_x).view(B, -1).sum(dim=1) / W
        pred_y = (prob * grid_y).view(B, -1).sum(dim=1) / H

        preds = torch.stack([pred_x, pred_y], dim=1)  # (B,2)
        return preds


def get_information_zone(img):
    """
    Extract specific regions from the input image.

    Args:
        img: np.array (original image)

    Returns:
        field_img: Cropped field image (main field)
        draw1_img: First cropped draw image
        draw2_img: Second cropped draw image
    """
    # Coordinates for cropping different parts of the image
    field_img = img[100:310, 65:275]  # Field coordinates
    draw1_img = img[5:55, 185:235]  # Draw 1 coordinates
    draw2_img = img[5:55, 260:310]  # Draw 2 coordinates

    return field_img, draw1_img, draw2_img


def cv2_to_pil(cv2_img):
    """
    Convert a cv2 image (BGR format) to a PIL Image in RGB format.
    """
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return pil_img


def make_predictions(field_img, draw1_img, draw2_img, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform inference using the trained model.

    Args:
        field_img: PIL Image (Field)
        draw1_img: PIL Image (Draw 1)
        draw2_img: PIL Image (Draw 2)
        model_path: str - Path to the trained model
        device: str - "cuda" or "cpu"

    Returns:
        (x1, y1, x2, y2): Tuple with predicted pixel coordinates.
    """
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert image to tensor
    ])
    
    # Apply transformations and add batch dimension
    field_img = transform(field_img).unsqueeze(0).to(device)  # (1, 3, 210, 210)
    draw1_img = transform(draw1_img).unsqueeze(0).to(device)  # (1, 3, 50, 50)
    draw2_img = transform(draw2_img).unsqueeze(0).to(device)  # (1, 3, 50, 50)

    # Load model
    model = DrawLocatorNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Perform inference
    with torch.no_grad():
        pred1_coords = model(field_img, draw1_img).cpu().numpy()[0]  # (2,)
        pred2_coords = model(field_img, draw2_img).cpu().numpy()[0]  # (2,)

    # Scale predictions to image dimensions (from normalized [0,1] to [0,210])
    pred_x1, pred_y1 = pred1_coords * 210
    pred_x2, pred_y2 = pred2_coords * 210

    # Adjust predictions relative to the original image (accounting for cropping)
    x1, x2 = int(pred_x1) + 65, int(pred_x2) + 65  # Offset by left_pixels (65)
    y1, y2 = int(pred_y1) + 100, int(pred_y2) + 100  # Offset by bot_pixels (100)

    return x1, y1, x2, y2


def run(img, model_path="model.pth"):
    """
    Full pipeline to process an image and get predictions.

    Args:
        img: np.array - Original image loaded using cv2
        model_path: str - Path to the trained model

    Returns:
        (x1, y1, x2, y2): Predicted coordinates
    """
    # Extract relevant regions
    field_img, draw1_img, draw2_img = get_information_zone(img)

    # Convert OpenCV images to PIL format
    field_img, draw1_img, draw2_img = cv2_to_pil(field_img), cv2_to_pil(draw1_img), cv2_to_pil(draw2_img)

    # Perform inference
    return make_predictions(field_img, draw1_img, draw2_img, model_path)


# Example
image_path = 'Your image path'
image = cv2.imread(image_path)
print(run(img=image))
