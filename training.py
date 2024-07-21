# train classifier


"""
Ce script utilise les données de points de repère des mains (extraites précédemment dans create_dataset) pour entraîner un modèle de classification. 
Le modèle est basé sur un classificateur Random Forest, et une fois entraîné, il est sauvegardé.
"""

import pickle
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# on charge le modèle
informations = pickle.load(open('./data.pickle', 'rb'))

# chargement de l'heure
now = datetime.now()

current_time = now.strftime("%Y-%m-%d %H_%M_%S")


# convertir car le type est List
data = np.asarray(informations['data'])
labels = np.asarray(informations['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)


print('score : ', score*100)


###### SAVE THE MODEL ######

f = open(f'sign_detector_model_{current_time}.p', 'wb')
pickle.dump({
    'model': model,
}, f)
f.close()
