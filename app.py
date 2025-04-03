# # Standard library imports
# import base64
# from functools import lru_cache
# from io import BytesIO
# from typing import List

# # Third-party library imports
# import cv2
# import numpy as np 
# from fastapi import FastAPI
# from fastapi import HTTPException
# from fastapi import Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# from ultralytics import YOLO
# import torch

# # Local imports
# from domain.domain import BoundingBox
# from domain.domain import RequestForObjectDetection
# from domain.domain import ResponseForObjectDetection


# from service.NatGeoCSIOceanClassificationService import NatGeoCSIOceanClassificationService
# CORSMiddleware

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS (Allow access from any origin)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to specific domains in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app = FastAPI() 

# # # Load YOLO model
# yolo_path = "artifacts/yolov10_exp22.pt"
# yolo_model = YOLO(yolo_path)



# @app.get("/")
# def read_root():
#     #return {"message": "Hello from MLdeploymentCSI"}
#     return {"message": yolo_model.names}


# def predict_csi_object_detection(image_base64: str, conf_threshold: float = 0.5) -> List[BoundingBox]:
#     """
#     Processes a single Base64-encoded image using YOLO object detection and returns the detected bounding boxes
#     for the original image.
    
#     Parameters:
#     - image_base64: Base64-encoded image.
#     - conf_threshold: Confidence threshold for filtering detections.
    
#     Returns:
#     - List of BoundingBox objects containing label, confidence, and bounding box coordinates for the original image.
#     """
#     try:
#         # Decode Base64 image
#         image_data_orginal = base64.b64decode(image_base64)
#         image_np = np.array(Image.open(BytesIO(image_data_orginal)))
#         img_rgb_orginal = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        
#         # Get original image dimensions for scaling
#         orig_height, orig_width = img_rgb_orginal.shape[:2]
        
#         # Resize image to 640x640 for YOLO input
#         img_rgb = cv2.resize(img_rgb_orginal, (640, 640))
        
#         # Run prediction
#         results = yolo_model.predict(img_rgb, conf=conf_threshold)
        
#         # List to store the detected objects (original image bounding boxes)
#         detections = []

#         # Extract detected objects
#         for result in results:
#             for box in result.boxes:
#                 conf = box.conf.item()
#                 if conf >= conf_threshold:
#                     # Get coordinates for the resized image
#                     x1, y1, x2, y2 = map(float, box.xyxy[0])

#                     # Scale coordinates back to original image size
#                     x1_orig = int(x1 * orig_width / 640)
#                     y1_orig = int(y1 * orig_height / 640)
#                     x2_orig = int(x2 * orig_width / 640)
#                     y2_orig = int(y2 * orig_height / 640)
                    
#                     # Add bounding boxes for the original image only
#                     detections.append(BoundingBox(
#                         label=result.names[int(box.cls)],
#                         x_min=x1_orig, y_min=y1_orig, x_max=x2_orig, y_max=y2_orig,
#                         confidence=round(conf, 2)
#                     ))
        
#         return detections
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# @app.post("/predict-object-detection/", response_model=ResponseForObjectDetection)
# async def detect_objects(request: RequestForObjectDetection):
#     """
#     FastAPI endpoint for object detection.
    
#     Parameters:
#     - request: RequestForObjectDetection containing Base64-encoded image.
    
#     Returns:
#     - ResponseForObjectDetection containing detected bounding boxes.
#     """
    
   
    
#     detections = predict_csi_object_detection( request.image_base64, conf_threshold=0.5)
    
#     return ResponseForObjectDetection(detections=detections)





# Standard library imports
import base64
from contextlib import contextmanager
from io import BytesIO
from typing import List

# Third-party imports
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torch.serialization import safe_globals
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


# Local imports
from domain.domain import (
    BoundingBox,
    RequestForObjectDetection,
    ResponseForObjectDetection
)

app = FastAPI()


app = FastAPI() 
# Enable CORS (Allow access from any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Load YOLO model
yolo_path = "artifacts/yolov10_exp22.pt"
yolo_model = YOLO(yolo_path)



@app.get("/")
def read_root():
    #return {"message": "Hello from MLdeploymentCSI"}
    
    return {"message": yolo_model.names}
#orginal
# def predict_csi_object_detection( image_base64: str, conf_threshold: float = 0.5) -> List[BoundingBox]:
#     """
#     Processes a single Base64-encoded image using YOLO object detection.
    
#     Parameters:
#     - model_path: Path to the trained YOLO model (.pt file).
#     - image_base64: Base64-encoded image.
#     - conf_threshold: Confidence threshold for filtering detections.
    
#     Returns:
#     - List of BoundingBox objects containing label, confidence, and bounding box coordinates.
#     """
#     try:
      
        
#         # Decode Base64 image
#         image_data_orginal = base64.b64decode(image_base64)
#         image_np = np.array(Image.open(BytesIO(image_data_orginal)))
#         img_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        
#         # Resize image to 640x640
#         img_rgb = cv2.resize(img_rgb, (640, 640))
        
#         # Run prediction
#         results = yolo_model.predict(img_rgb, conf=conf_threshold)
        
#         # List to store detected objects
#         detections = []
        
#         # Extract detected objects
#         for result in results:
#             for box in result.boxes:
#                 conf = box.conf.item()
#                 if conf >= conf_threshold:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     detections.append(BoundingBox(
#                         label=result.names[int(box.cls)],
#                         x_min=x1, y_min=y1, x_max=x2, y_max=y2,
#                         confidence=round(conf, 2)
#                     ))
        
#         return detections
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# from fastapi import HTTPException
# import base64
# import cv2
# import numpy as np
# from PIL import Image
# from typing import List
# from io import BytesIO
# from pydantic import BaseModel

# # Define the BoundingBox class to structure the bounding box data
# class BoundingBox(BaseModel):
#     label: str  # Label for the detected object
#     x_min: int  # Minimum x-coordinate of the bounding box
#     y_min: int  # Minimum y-coordinate of the bounding box
#     x_max: int  # Maximum x-coordinate of the bounding box
#     y_max: int  # Maximum y-coordinate of the bounding box
#     confidence: float  # Confidence score of the detection

def predict_csi_object_detection(image_base64: str, conf_threshold: float = 0.5) -> List[BoundingBox]:
    """
    Processes a single Base64-encoded image using YOLO object detection and returns the detected bounding boxes
    for the original image.
    
    Parameters:
    - image_base64: Base64-encoded image.
    - conf_threshold: Confidence threshold for filtering detections.
    
    Returns:
    - List of BoundingBox objects containing label, confidence, and bounding box coordinates for the original image.
    """
    try:
        # Decode Base64 image
        image_data_orginal = base64.b64decode(image_base64)
        image_np = np.array(Image.open(BytesIO(image_data_orginal)))
        img_rgb_orginal = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        
        # Get original image dimensions for scaling
        orig_height, orig_width = img_rgb_orginal.shape[:2]
        
        # Resize image to 640x640 for YOLO input
        img_rgb = cv2.resize(img_rgb_orginal, (640, 640))
        
        # Run prediction
        results = yolo_model.predict(img_rgb, conf=conf_threshold)
        
        # List to store the detected objects (original image bounding boxes)
        detections = []

        # Extract detected objects
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                if conf >= conf_threshold:
                    # Get coordinates for the resized image
                    x1, y1, x2, y2 = map(float, box.xyxy[0])

                    # Scale coordinates back to original image size
                    x1_orig = int(x1 * orig_width / 640)
                    y1_orig = int(y1 * orig_height / 640)
                    x2_orig = int(x2 * orig_width / 640)
                    y2_orig = int(y2 * orig_height / 640)
                    
                    # Add bounding boxes for the original image only
                    detections.append(BoundingBox(
                        label=result.names[int(box.cls)],
                        x_min=x1_orig, y_min=y1_orig, x_max=x2_orig, y_max=y2_orig,
                        confidence=round(conf, 2)
                    ))
        
        return detections
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict-object-detection/", response_model=ResponseForObjectDetection)
async def detect_objects(request: RequestForObjectDetection):
    """
    FastAPI endpoint for object detection.
    
    Parameters:
    - request: RequestForObjectDetection containing Base64-encoded image.
    
    Returns:
    - ResponseForObjectDetection containing detected bounding boxes.
    """
    
   
    
    detections = predict_csi_object_detection( request.image_base64, conf_threshold=0.5)
    
    return ResponseForObjectDetection(detections=detections)

