#Grayson Beamesderfer
#9/29/25
#this program is an ai chatbot but you talk to it using sign language

import cv2
import customtkinter as ctk
import openai
import mediapipe as mp

############################# HAND TRACKING SET UP FROM TUTORIAL ####################################

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

#set up hand tracking and drawing
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model = mp_hands.Hands()

while True:
    #read a frame from the camera
    success, img = cam.read()
    
    if success:
        RGB_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = model.process(RGB_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                print(hand_landmark)
                mp_drawing.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("captured image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and writer objects
cv2.destroyAllWindows()