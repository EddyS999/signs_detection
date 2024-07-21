# post processing

"""
Ce script utilise la bibliothèque MediaPipe pour extraire les points de repère des mains (hand landmarks) à partir des images capturées dans collect_img. 
Il compile ensuite ces informations et les sauvegarde dans un fichier pickle l'entraînement du modèle.

"""


import pickle
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import cv2


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data'

data = []
label = []
for i in os.listdir(DATA_DIR):
    for j in os.listdir(os.path.join(DATA_DIR, i)):
        # On va stocker toutes les informations de toutes les images
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, i, j))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for k in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[k].x
                    y = hand_landmarks.landmark[k].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            label.append(i)  # contient les differents mots


f = open('data.pickle', 'wb')
pickle.dump({
    'data': data,
    'labels': label,
}, f)
f.close()

# mp_drawing.draw_landmarks(
#     img_rgb,
#     hand_landmarks,
#     mp_hands.HAND_CONNECTIONS,
#     mp_drawing_styles.get_default_hand_landmarks_style(),
#     mp_drawing_styles.get_default_hand_connections_style()
# )

# plt.figure()
# plt.imshow(img_rgb)

# plt.show()
