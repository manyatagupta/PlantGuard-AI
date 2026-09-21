import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an image for MobileNetV2.
    
    Args:
        image: PIL Image object or path to image.
        target_size: Tuple representing the target size for the model.
        
    Returns:
        numpy array: Preprocessed image batch ready for prediction.
    """
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
        
    # Convert to RGB in case it has an alpha channel or is grayscale
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Resize the image
    img = img.resize(target_size)
    
    # Convert image to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess input (MobileNetV2 expects values between -1 and 1)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array
