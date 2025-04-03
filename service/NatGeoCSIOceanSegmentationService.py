#still in development stage
import pickle
import base64
import io
import numpy as np
from PIL import Image
from typing import List
from domain.domain import RequestForImageSegmentation, ResponseForImageSegmentation, SegmentationMask

class NatGeoCSIOceanSegmentationService:
    def __init__(self):
        # Define the path to the segmentation model artifact
        self.path_model = "artifacts/segmentation_model.pkl"
        
        # Load the model
        self.model = self.load_artifact(self.path_model)

    def load_artifact(self, path_to_artifact):
        """Load artifact from a pickle file."""
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        return artifact

    def preprocess_input(self, request: RequestForImageSegmentation) -> np.ndarray:
        """Process and prepare the input image for segmentation."""
        # Decode the Base64 image
        image_data = self.decode_image(request.image_base64)
        
        # Transform the image data to fit the model's expected input format
        preprocessed_data = self.transform_image_data(image_data)
        
        return preprocessed_data

    def decode_image(self, image_base64: str) -> Image:
        """Decode a base64-encoded image string into an image format usable by the model."""
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    def transform_image_data(self, image: Image) -> np.ndarray:
        """Transform the decoded image data for model compatibility."""
        image = image.resize((224, 224))  # Adjust based on model input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return image_array

    def segment_image(self, request: RequestForImageSegmentation) -> ResponseForImageSegmentation:
        """Segment the image and return a mask for each detected class or object."""
        input_data = self.preprocess_input(request)
        
        # Perform segmentation using the loaded model
        mask_predictions = self.model.predict(input_data[np.newaxis, ...])[0]  # Add batch dimension
        
        # Process the segmentation masks and encode them in Base64
        masks = self.process_masks(mask_predictions)
        
        return ResponseForImageSegmentation(segmentation_masks=masks)

    def process_masks(self, masks: np.ndarray) -> List[SegmentationMask]:
        """Process raw masks and convert them to a list of Base64-encoded segmentation masks."""
        mask_list = []
        
        # Iterate over each mask layer (e.g., for each class)
        for mask in masks:
            # Convert mask to image and encode in Base64
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to binary image format
            buffered = io.BytesIO()
            mask_image.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Append to list as SegmentationMask instance
            mask_list.append(SegmentationMask(mask=mask_base64))
        
        return mask_list



