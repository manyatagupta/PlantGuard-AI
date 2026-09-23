import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utils.preprocessing import preprocess_image

class PlantDiseasePredictor:
    def __init__(self, model_path="models/plant_disease_model.keras", class_names_path="models/class_names.txt"):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        
    def load_model(self):
        """Loads the trained model and class names if they exist."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
            
        if not os.path.exists(self.class_names_path):
             raise FileNotFoundError(f"Class names not found at {self.class_names_path}. Please train the model first.")
             
        self.model = tf.keras.models.load_model(self.model_path)
        
        with open(self.class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
            
        return True
        
    def predict(self, image):
        """
        Predicts the disease class of a plant leaf image.
        
        Args:
            image: PIL Image or file path.
            
        Returns:
            dict: Containing 'prediction', 'confidence', 'is_healthy', raw probabilities, and heatmap.
        """
        if self.model is None:
            self.load_model()
            
        # Preprocess image
        processed_img = preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class_name = self.class_names[predicted_class_idx]
        
        # Generate Grad-CAM heatmap
        heatmap = self.make_gradcam_heatmap(processed_img, predicted_class_idx)
        
        # Determine if healthy (assuming 'healthy' is in the class name)
        is_healthy = 'healthy' in predicted_class_name.lower()
        
        return {
            'predicted_class': predicted_class_name,
            'confidence': confidence,
            'is_healthy': is_healthy,
            'all_probabilities': {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))},
            'heatmap': heatmap
        }
        
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """Generates a Grad-CAM heatmap for the given image and class."""
        try:
            # Find the last convolutional layer or the nested base model
            target_layer = None
            for layer in reversed(self.model.layers):
                # Check for a nested base model (e.g., EfficientNet, MobileNet)
                if isinstance(layer, tf.keras.Model):
                    target_layer = layer
                    break
                # Alternatively, check for a standalone convolutional layer
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer = layer
                    break
            
            if target_layer is None:
                return None
                
            # Create a model that maps the input image to the activations
            # of the target layer as well as the output predictions
            grad_model = Model(
                [self.model.inputs], 
                [target_layer.output, self.model.output]
            )

            # Compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            # Gradient of the output neuron w.r.t. the output feature map
            grads = tape.gradient(class_channel, last_conv_layer_output)

            # Mean intensity of the gradient over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Multiply each channel in the feature map array by "how important this channel is"
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Normalize the heatmap between 0 and 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
        except Exception as e:
            print(f"Failed to generate Grad-CAM: {e}")
            return None

# Example usage (for testing purposes)
if __name__ == "__main__":
    predictor = PlantDiseasePredictor()
    try:
        predictor.load_model()
        print(f"Model loaded successfully. Classes: {predictor.class_names}")
    except FileNotFoundError as e:
        print(e)
