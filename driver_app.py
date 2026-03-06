import cv2
import dlib
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
import pyttsx3
import pygame

# ---------------- SOUND ----------------
pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")  # your external audio file
pygame.mixer.music.set_volume(1.0)

# ---------------- SETTINGS ----------------
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 5   # faster detection

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

# ---------------- LOAD MODEL ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks .dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# ---------------- GUI CLASS ----------------
class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection System")
        self.root.geometry("900x700")
        self.running = False
        self.counter = 0
        self.alarm_on = False

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.status_label = tk.Label(root, text="Status: Waiting",
                                     font=("Arial", 18), fg="blue")
        self.status_label.pack(pady=10)

        self.start_btn = tk.Button(root, text="Start Detection",
                                   command=self.start, width=20, bg="green", fg="white")
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(root, text="Stop Detection",
                                  command=self.stop, width=20, bg="red", fg="white")
        self.stop_btn.pack(pady=5)

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        pygame.mixer.music.stop()
        self.status_label.config(text="Status: Stopped", fg="black")

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
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

                    if ear < EYE_AR_THRESH:
                        self.counter += 1

                        if self.counter >= EYE_AR_CONSEC_FRAMES:
                            self.status_label.config(text="Status: DROWSY!",
                                                     fg="red")

                            if not self.alarm_on:
                                self.alarm_on = True
                                speak_once()
                                pygame.mixer.music.play(-1)  # loop sound

                    else:
                        self.counter = 0

                        if self.alarm_on:
                            pygame.mixer.music.stop()

                        self.alarm_on = False
                        self.status_label.config(text="Status: Awake",
                                                 fg="green")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_frame)

# ---------------- RUN APP ----------------
root = tk.Tk()
app = DrowsinessApp(root)
root.mainloop()