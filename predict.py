import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utils.preprocessing import preprocess_image

def parse_crop_and_disease(class_name: str) -> tuple:
    """
    Parses raw dataset class string into user-friendly crop and disease strings.
    Example: 'Apple___Apple_scab' -> ('Apple', 'Apple Scab')
    """
    clean = class_name.replace(" - Copy", "").strip()
    if "___" in clean:
        parts = clean.split("___", 1)
        crop = parts[0].replace("_(maize)", "").replace("__bell", "").replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    elif clean.startswith("Tomato_"):
        crop = "Tomato"
        disease = clean[len("Tomato_"):].lstrip("_").replace("_", " ").strip()
    else:
        parts = clean.split("_", 1)
        crop = parts[0]
        disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
        
    if not disease or disease.lower() == "healthy":
        disease = "Healthy"
    return crop, disease


class PlantDiseasePredictor:
    def __init__(self, model_path="models/plant_disease_model.keras", class_names_path="models/class_names.txt"):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        self.has_gradcam = False
        self.feature_extractor = None
        self.classifier_model = None
        
    def load_model(self):
        """Loads the trained model and class names if they exist."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
            
        if not os.path.exists(self.class_names_path):
             raise FileNotFoundError(f"Class names not found at {self.class_names_path}. Please train the model first.")
             
        self.model = tf.keras.models.load_model(self.model_path)
        
        with open(self.class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
            
        self._setup_gradcam()
        return True

    def _setup_gradcam(self):
        """Preconfigures sub-models for Grad-CAM explainability."""
        try:
            base_model = None
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.Model) and "sequential" not in layer.name.lower():
                    base_model = layer
                    break

            if base_model is not None:
                last_conv = None
                try:
                    last_conv = base_model.get_layer("top_conv")
                except Exception:
                    for l in reversed(base_model.layers):
                        if "conv" in l.name.lower() or "conv" in type(l).__name__.lower():
                            last_conv = l
                            break

                if last_conv is not None:
                    self.feature_extractor = tf.keras.Model(
                        inputs=base_model.input,
                        outputs=[last_conv.output, base_model.output]
                    )

                    classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
                    x = classifier_input
                    base_idx = self.model.layers.index(base_model)
                    for layer in self.model.layers[base_idx + 1:]:
                        x = layer(x)
                    self.classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)
                    self.has_gradcam = True
                    return
        except Exception as e:
            print(f"Grad-CAM setup note: {e}")
        self.has_gradcam = False
        
    def is_loaded(self):
        """Returns True if the neural network model is loaded and ready."""
        return self.model is not None

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
        if not getattr(self, 'has_gradcam', False):
            return None
        try:
            img_tensor = tf.cast(img_array, tf.float32)
            with tf.GradientTape() as tape:
                conv_out, base_out = self.feature_extractor(img_tensor)
                tape.watch(conv_out)
                preds = self.classifier_model(base_out)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_0 = conv_out[0]
            heatmap = conv_0 @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
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
