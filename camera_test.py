import cv2
import dlib

# Load face detector
detector = dlib.get_frontal_face_detector()

# 👉 ADD THIS LINE HERE
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # here you calculate EAR
        # here you check drowsiness
