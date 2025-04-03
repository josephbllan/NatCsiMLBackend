from pydantic import BaseModel
from typing import List, Union



# Request and response models for image classification
class RequestForImageClassification(BaseModel):
    image_base64: str  # Base64-encoded image string 

class ResponseForImageClassification(BaseModel):
    classification_result: Union[str, List[str]]  # Can be a single label or a list of top labels
    confidence: float  # Confidence score 

# Request and response models for object detection
class RequestForObjectDetection(BaseModel):
    image_base64: str  # Base64-encoded image string 

class BoundingBox(BaseModel):
    label: str  # Label for the detected object
    x_min: int  # Minimum x-coordinate of the bounding box
    y_min: int  # Minimum y-coordinate of the bounding box
    x_max: int  # Maximum x-coordinate of the bounding box
    y_max: int  # Maximum y-coordinate of the bounding box
    confidence: float  # Confidence score of the detection 

class ResponseForObjectDetection(BaseModel):
    detections: List[BoundingBox]  # List of bounding boxes with labels
    
    
class RequestForImageSegmentation(BaseModel):
    image_base64: str  # Base64-encoded input image

class SegmentationMask(BaseModel):
    mask: str  # Base64-encoded segmentation mask image (e.g., in grayscale or binary format)

class ResponseForImageSegmentation(BaseModel):
    segmentation_masks: List[SegmentationMask]  # List of segmentation masks for each detected object or class