import cv2 as cv
from ultralytics import YOLO
from utils import extract_results

modelPath = "Model"  # Enter your model path
source = "Source"  # Enter your source path (for camera add a number like 0 or 1)

# Load the model
model = YOLO(modelPath)

# Open cap
cap = cv.VideoCapture(source)

# Loop through entire cap
while True:
    ret, frame = cap.read()  # Read the cap and unpack frame and ret
    
    if not ret:
        break  # Break if true
    
    # Run the detection model on the frame
    prediction = model.predict(frame, stream=True, imgsz=640)
    
    # Extract information about the results
    res = extract_results(prediction, frame)
    
    # Print the results
    print(res)