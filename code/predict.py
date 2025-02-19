import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Load the YOLO model from a pre-trained file 'best.pt'
model = YOLO('best.pt')

# Define a function to capture mouse movements and print the coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window named 'RGB' and set the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Start capturing video from the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Read class names from a file 'coco.txt' and store them in a list
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)
count = 0

# Start an infinite loop to read frames from the video capture
while True:
    ret, frame = cap.read()  # Read a frame from the video capture
    count += 1  # Increment the frame counter
    if count % 3 != 0:  # Skip frames to reduce the load on the system
        continue

    frame = cv2.resize(frame, (1020, 500))  # Resize the frame

    results = model.predict(frame)  # Use the YOLO model to make predictions on the frame
    a = results[0].boxes.data  # Extract the detected boxes data
    px = pd.DataFrame(a).astype("float")  # Convert the data to a pandas DataFrame

    for index, row in px.iterrows():  # Iterate over each detected object
        x1 = int(row[0])  # Get the top-left x coordinate
        y1 = int(row[1])  # Get the top-left y coordinate
        x2 = int(row[2])  # Get the bottom-right x coordinate
        y2 = int(row[3])  # Get the bottom-right y coordinate
        d = int(row[5])  # Get the class ID
        c = class_list[d]  # Get the class name from the class list
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the detected object
        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)  # Put the class name text above the rectangle
    
    cv2.imshow("RGB", frame)  # Show the frame in the 'RGB' window
    if cv2.waitKey(1) & 0xFF == 27:  # Exit the loop if the 'Esc' key is pressed
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
