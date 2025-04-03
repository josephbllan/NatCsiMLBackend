#still in development stage
import pickle
import numpy as np
import base64
import io
from PIL import Image
from domain.domain import RequestForObjectDetection, ResponseForObjectDetection

class NatGeoCSIOceanObjectDetectionService:
    def __init__(self):
        # Define paths to the object detection model artifact
        self.path_model = "artifacts/object_detection_model.pkl"
        
        # Load the model
        self.model = self.load_artifact(self.path_model)

    def load_artifact(self, path_to_artifact):
        """Load artifact from a pickle file."""
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        return artifact

    def preprocess_input(self, request: RequestForObjectDetection) -> np.ndarray:
        """Process and prepare the input image for object detection."""
        # Decode the Base64 image and perform necessary transformations
        image_data = self.decode_image(request.image_base64)
        
        # Transform the image data to fit the model's expected input format
        preprocessed_data = self.transform_image_data(image_data)
        
        return preprocessed_data

    def decode_image(self, image_base64: str) -> Image:
        """Decode a base64-encoded image string into an image format usable by the model."""
        # Decode the base64 string
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    def transform_image_data(self, image: Image) -> np.ndarray:
        """Transform the decoded image data for model compatibility."""
        # Resize the image (assuming model expects 224x224 images)
        image = image.resize((224, 224))
        # Convert to a NumPy array
        image_array = np.array(image)  
        return image_array

    def detect_objects(self, request: RequestForObjectDetection) -> ResponseForObjectDetection:
        """Detect objects in the image and return their bounding boxes and labels."""
        input_data = self.preprocess_input(request)
        
        # Perform object detection using the loaded model
        predictions = self.model.predict(input_data[np.newaxis, ...])  # Add batch dimension
        
        # Process predictions to extract bounding boxes and labels
        response = self.process_predictions(predictions)
        
        return response

    def process_predictions(self, predictions) -> ResponseForObjectDetection:
        """Process raw predictions to extract bounding boxes and labels."""
        # Extract bounding boxes and labels from predictions (implementation depends on the model output format)
        bboxes = []  # List to hold bounding boxes
        labels = []  # List to hold labels
        
        for prediction in predictions:
            # Assume prediction contains bounding box coordinates and class indices
            # Implement the logic based on your model's output format
            for det in prediction:
                bbox = det[:4]  # Assuming first four entries are x1, y1, x2, y2
                label_index = int(det[4])  # Assuming fifth entry is the label index
                bboxes.append(bbox)
                labels.append(label_index)

        # Create a response object with bounding boxes and corresponding labels
        response = ResponseForObjectDetection(bounding_boxes=bboxes, labels=labels)
        return response
    

    
