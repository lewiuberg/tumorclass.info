import base64
import io
import os

import numpy as np

# for our model
import tensorflow as tf

# to retrieve and send back data
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model


CLASSES = ["normal", "lgg", "hgg"]
MODELS_PATH = (
    "/Users/lewiuberg/Documents/Data Science/Projects/tumorclass.info/models/"
)
MODEL_NAME = "cnn_densenet121"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

app = Flask(__name__)


def get_model(model_name):
    model = load_model(os.path.join(MODELS_PATH, f"{model_name}_model.h5"))
    print("[+] model loaded")
    return model


def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded


# preprocess image before sending it to the model
def preprocess(decoded):
    # resize and convert in RGB in case image is in RGBA
    pil_image = (
        Image.open(io.BytesIO(decoded))
        .resize((224, 224), Image.LANCZOS)
        .convert("RGB")
    )
    image = np.asarray(pil_image)
    batch = np.expand_dims(image, axis=0)

    return batch


# def decode_request(req):
#     # print keys in req and pause
#     # img = load_img(req["image"], target_size=(IMAGE_WIDTH, IMAGE_WIDTH))
#     encoded = req["image"]
#     decoded = base64.b64decode(encoded)
#     # image to array
#     # img = Image.open(io.BytesIO(decoded))
#     # expand_dims
#     img_array = img_to_array(decoded)
#     # img = Image.open(io.BytesIO(decoded))
#     # img_array = img_to_array(encoded)
#     # img_array = expand_dims(encoded, 0)  # Create batch axis
#     return img_array


# # decode the imaeg coming from the request
# def decode_request(req):
#     encoded = req["image"]
#     decoded = base64.b64decode(encoded)
#     return decoded


model = get_model(MODEL_NAME)


@app.route("/", methods=["POST"])
def index():
    # serve static html file static/index.html
    return render_template("static/index.html")


@app.route("/predict", methods=["POST"])
def predict():

    req = request.get_json(force=True)
    image = decode_request(req)
    batch = preprocess(image)

    predictions = model.predict(batch)

    response = {
        "classification": CLASSES[np.argmax(predictions[0])].upper(),
        "confidence": f"{np.max(predictions[0])*100:.2f}%",
        CLASSES[0]: f"{predictions[0][0]*100:.2f}%",
        CLASSES[1]: f"{predictions[0][1]*100:.2f}%",
        CLASSES[2]: f"{predictions[0][2]*100:.2f}%",
    }

    print("[+] results {}".format(response))

    return jsonify(response)  # return it as json


# predictor(
#     path="/Users/lewiuberg/Documents/Data Science/Projects/tumorclass.info/data/dataset/test/lgg/21.jpg",
#     image_width=224,
#     image_height=224,
#     model=get_model(),
#     classes=CLASSES,
# )


#     # create a variable named app
# app = Flask(__name__)

# # create our model
# IMG_SHAPE = (224, 224, 3)

# def get_model():
#     model = ResNet50(
#         include_top=True, weights="imagenet", input_shape=IMG_SHAPE
#     )
#     print("[+] model loaded")
#     return model

# # decode the imaeg coming from the request
# def decode_request(req):
#     encoded = req["image"]
#     decoded = base64.b64decode(encoded)
#     return decoded

# # preprocess image before sending it to the model
# def preprocess(decoded):
#     # resize and convert in RGB in case image is in RGBA
#     pil_image = (
#         Image.open(io.BytesIO(decoded))
#         .resize((224, 224), Image.LANCZOS)
#         .convert("RGB")
#     )
#     image = np.asarray(pil_image)
#     batch = np.expand_dims(image, axis=0)

#     return batch

# # load model so it's in memory and not loaded each time there is a request
# model = get_model()

# # function predict is called at each request
# @app.route("/predict", methods=["POST"])
# def predict():
#     print("[+] request received")
#     # get the data from the request and put ir under the right format
#     req = request.get_json(force=True)
#     image = decode_request(req)
#     batch = preprocess(image)
#     # actual prediction of the model
#     prediction = model.predict(batch)
#     # get the label of the predicted class. get result anf confidence
#     result = decode_predictions(prediction)
#     print("LEWI\n\n")
#     print(result)
#     print("LEWI\n\n")
#     # get get [1] and [2] of result[0][0] ina tuple comprehension
#     top_label = (
#         result[0][0][1],
#         str(result[0][0][2]),
#     )
#     print("LEWI\n\n")
#     print(top_label)
#     print("LEWI\n\n")
#     # top_label = decode_predictions(prediction, top=1)[0][0][1]
#     # top_label = [
#     #     (i[1], str(i[2])) for i in decode_predictions(prediction)[0][0]
#     # ]
#     # create the response as a dict
#     response = {"prediction": top_label}
#     print("[+] results {}".format(response))

#     return jsonify(response)  # return it as json
