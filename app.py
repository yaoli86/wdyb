from flask import Flask
import cv2
import uuid
import base64
import requests
import numpy as np
from keras.models import model_from_json
from flask import request, jsonify
import json

FACE_CASCADE = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
emotions = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# load json and create model arch
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model.h5')
model._make_predict_function()

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'

if __name__ == '__main__':
    print('Start ^^')
    app.run()
