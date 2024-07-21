"""
fichier qui gère la logique de l'application en traitant les requêtes et les actions de l'utilisateur. 
Elle interagit également avec les templates pour présenter les informations à l'utilisateur et avec le modèle pour accéder aux données ou effectuer des opérations de traitement. 
Dans notre cas, l’ensemble des fonctions qui gère l’interaction avec le modèle de deep learning sont dans le fichier view.py

"""


import cv2
from typing import Any
from django.shortcuts import render, HttpResponse
from django.http import HttpRequest, JsonResponse
import base64
import json
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import openai
import numpy as np
import mediapipe as mp
from chatapp.model_loader import model
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        module='google.protobuf.symbol_database')


class PredictView(View):
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            image_data = data['image']
            image_data = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            prediction = get_prediction_from_model(image)
            # renvoie de la prediction dans l'async getPrediction
            return JsonResponse({'prediction': prediction})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


def get_prediction_from_model(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           min_detection_confidence=0.3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for k in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[k].x
                y = hand_landmarks.landmark[k].y
                data_aux.append(x)
                data_aux.append(y)

        # On récupère le model entrainé pour effectuer une prédiction sur les valeurs x et y
        prediction = model.predict([np.asarray(data_aux)])
        return prediction[0]


def correct_sentence(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
                "content": "You are a helpful assistant that corrects sentences."},
            {"role": "user", "content": f"Cette phrase manque certains mots pour etre comprise, corrige juste cette phrase sans dire autre chose: '{message}'"},
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    corrected_sentence = response['choices'][0]['message']['content'].strip()
    return corrected_sentence


def ask_openai(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer


def get_suggestions(message):
    corrected_message = correct_sentence(message)
    suggestions = [
        corrected_message
    ]
    return suggestions


def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        corrected_message = correct_sentence(message)
        response = ask_openai(corrected_message)
        suggestions = get_suggestions(message)
        return JsonResponse({
            'original_message': message,
            'corrected_message': corrected_message,
            'response': response,
            'suggestions': suggestions
        })
    return render(request, "chatbot.html")


def base(request):
    return render(request, "base.html")
