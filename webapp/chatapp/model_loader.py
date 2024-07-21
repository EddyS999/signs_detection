import pickle
import os
import pickle


def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'sign_detector_model.p')
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model']


model = load_model()
