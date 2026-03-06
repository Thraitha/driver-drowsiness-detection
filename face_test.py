import cv2
import dlib
import numpy as np
import pyttsx3
from scipy.spatial import distance as dist
import pygame

# ---------------- PYGAME SOUND ----------------
pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")  # Make sure file exists
pygame.mixer.music.set_volume(1.0)

# ---------------- SETTINGS ----------------
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
ALARM_ON = False

# ---------------- VOICE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_once():
    engine.say("Wake up driver!")
    engine.runAndWait()

# ---------------- EAR FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------- LOAD MODELS ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks .dat")  
# ❗ removed extra space in filename

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        # -------- DROWSINESS CHECK --------
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3)

                if not ALARM_ON:
                    ALARM_ON = True

                    # Step 1: Speak once
                    speak_once()

                    # Step 2: Play music continuously
                    pygame.mixer.music.play(-1)

        else:
            COUNTER = 0

            if ALARM_ON:
                pygame.mixer.music.stop()

            ALARM_ON = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()