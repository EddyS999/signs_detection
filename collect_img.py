import os
import cv2

"""
Ce script permet de capturer des images à partir d'une caméra vidéo 
et de les sauvegarder dans des dossiers spécifiques en fonction de mots définis.
Il constitue la premiere étape.
"""

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


mots = ['IloveYou', "Hello", "No", "Yes"]
dataset_size = 200

# Video capture a changer en fonction de la machine
cap = cv2.VideoCapture(1)

for i in mots:
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))
    print('Collecte des données pour :', i)
    done = False

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Z".', (100, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('z'):
            break
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(
            i), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()