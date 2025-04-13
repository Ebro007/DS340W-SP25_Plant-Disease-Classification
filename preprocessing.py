import cv2
import numpy as np
from enhancements import applyCLAHE, applyHFEFilter




def preprocessing(image):
    """
    Custom preprocessing function to apply CLAHE and HFE filtering.
    
    Args:
        image: A NumPy array representing the image in RGB format with pixel values in [0, 255].
    
    Returns:
        The preprocessed image as a float32 NumPy array scaled to [0, 1].
    """
    # Convert RGB to BGR for OpenCV functions
    image_bgr = image[..., ::-1]
    
    # Apply CLAHE filtering
    image_clahe = applyCLAHE(image_bgr)
    
    # Apply High Frequency Emphasis (HFE) filtering
    image_hfe = applyHFEFilter(image_clahe)
    
    # Convert back to RGB (swap channels back)
    image_rgb = image_hfe[..., ::-1]
    
    # Convert image to float32 and scale to [0, 1]
    image_float = image_rgb.astype("float32") / 255.0
    
    return image_float





