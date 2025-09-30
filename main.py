#Grayson Beamesderfer
#9/29/25
#this program is an ai chatbot but you talk to it using sign language

import cv2
import customtkinter as ctk
import openai
from ultralytics import YOLO

model = YOLO("Misc\\yolov8n.pt")

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Misc\\output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Hand Tracking", annotated)

    # Write the frame to the output file
    out.write(frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()