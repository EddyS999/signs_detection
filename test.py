"""
Ce script utilise un modèle de classification préalablement entraîné pour détecter et reconnaître des gestes de la main en temps réel à partir d'une capture vidéo. 
Il utilise MediaPipe pour l'extraction des points de repère des mains et OpenCV pour la capture vidéo et l'affichage.
"""

import cv2
import mediapipe as mp
import pickle
import numpy as np
import warnings
from training import current_time

warnings.filterwarnings('ignore', category=UserWarning,
                        module='google.protobuf.symbol_database')

model_dict = pickle.load(open(f'./sign_detector_model_{current_time}.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_dict = {0: 'Hello', 1: 'IloveYou', 2: 'No', 3: 'Yes'}
last_prediction = None  # Variable pour stocker la dernière prédiction

while True:
    data_aux = []
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for k in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[k].x
                y = hand_landmarks.landmark[k].y
                data_aux.append(x)
                data_aux.append(y)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_word = prediction[0]

        if isinstance(predicted_word, str):
            current_prediction = predicted_word
        else:
            current_prediction = label_dict.get(int(predicted_word), "Unknown")

        if current_prediction != last_prediction:
            print(f"Model prediction: {current_prediction}")
            last_prediction = current_prediction

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
