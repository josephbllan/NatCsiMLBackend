
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
from typing import List
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.lite.python.interpreter import Interpreter

from domain.domain import RequestForImageClassification, ResponseForImageClassification

class NatGeoCSIOceanClassificationService:
    def __init__(self, model_path: str, model_type: str = 'pkl', class_names: List[str] = None):
        self.model_path = model_path
        self.model_type = model_type
        self.class_names = class_names or ["Background material", "Cellulose filter fibre", "Cotton", "Glass filter fibre"]
        self.model = self._load_model()  

    def _load_model(self):
        """Load the model based on type ('pkl' or 'tflite')."""
        if self.model_type == 'pkl':
            # Load the Pickle model (typically a scikit-learn or Keras model)
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        elif self.model_type == 'tflite':
            # Load the TensorFlow Lite model
            interpreter = Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            raise ValueError("Unsupported model type. Use 'pkl' or 'tflite'.")

    def _preprocess_image(self, image_base64: str) -> np.ndarray:
        """Preprocess the image from base64 to the model's expected format."""
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(BytesIO(image_data)).resize((224, 224))
            img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")


    def _predict_with_pkl(self, img_array: np.ndarray) -> np.ndarray:
        """Make prediction using a pickle model."""
        return self.model.predict(img_array)

    def _predict_with_tflite(self, img_array: np.ndarray) -> np.ndarray:
        """Make prediction using a TensorFlow Lite model."""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.model.set_tensor(input_details[0]['index'], img_array)
        self.model.invoke()
        return self.model.get_tensor(output_details[0]['index'])

    def classify_image(self, request: RequestForImageClassification) -> ResponseForImageClassification:
        """Classify an image and return the top prediction."""
        img_array = self._preprocess_image(request.image_base64)
        # Predict using the appropriate model type
        predictions = self._predict_with_pkl(img_array) if self.model_type == 'pkl' else self._predict_with_tflite(img_array)
        
        # Get the top prediction (class with the highest probability)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        
        # Return the top predicted class and confidence
        return ResponseForImageClassification(
            classification_result=predicted_class,
            confidence=confidence
        )

    def classify_image_top_n(self, request: RequestForImageClassification, n: int) -> ResponseForImageClassification:
        """Classify an image and return the top N predictions."""
        img_array = self._preprocess_image(request.image_base64)
        # Predict using the appropriate model type
        predictions = self._predict_with_pkl(img_array) if self.model_type == 'pkl' else self._predict_with_tflite(img_array)
        
        # Get the top N predictions
        top_n_indices = predictions[0].argsort()[-n:][::-1]
        top_n_classes = [self.class_names[i] for i in top_n_indices]
        return ResponseForImageClassification(
            classification_result=top_n_classes,
            confidence=float(predictions[0][top_n_indices[0]])  # confidence for top prediction
        )


