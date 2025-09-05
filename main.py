import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 'yolov5s' is the small version

def detect_objects(frame):
    """Detect objects in a frame using YOLOv5 and return annotated frame"""
    # Convert BGR (OpenCV) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(rgb_frame)
    
    # Parse results
    detections = results.pandas().xyxy[0]  # DataFrame with: xmin, ymin, xmax, ymax, confidence, class, name
    
    # Draw bounding boxes and labels
    for _, detection in detections.iterrows():
        label = f"{detection['name']} {detection['confidence']:.2f}"
        xmin, ymin, xmax, ymax = detection[['xmin','ymin','xmax','ymax']].astype(int)
        
        # Draw rectangle
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw label background
        cv2.rectangle(frame, (xmin, ymin-20), (xmin+len(label)*10, ymin), (0, 255, 0), -1)
        
        # Put text
        cv2.putText(frame, label, (xmin, ymin-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

# ======== IMAGE DETECTION ========
def detect_image(image_path):
    """Process single image"""
    img = cv2.imread(image_path)
    result_img = detect_objects(img)
    cv2.imshow('YOLO Object Detection', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ======== VIDEO DETECTION ========
def detect_video(video_path=0):
    """Process video stream (0 for webcam)"""
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects
        result_frame = detect_objects(frame)
        
        # Display
        cv2.imshow('YOLO Real-time Detection', result_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose detection mode:
    # detect_video()  # For Webcam
     detect_video("videoplayback.mp4")  # For video file
    # detect_image("img.jpg")  # For Image file