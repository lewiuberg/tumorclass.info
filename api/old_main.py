import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from tensorflow.keras.models import load_model


CLASSES = ["normal", "lgg", "hgg"]
MODELS_PATH = (
    "/Users/lewiuberg/Documents/Data Science/Projects/tumorclass.info/models/"
)
MODEL_NAME = "cnn_densenet121"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# app = Flask(__name__)


def get_model(model_name):
    model = load_model(os.path.join(MODELS_PATH, f"{model_name}_model.h5"))
    print("[+] model loaded")
    return model


def decode_request(req):
    encoded = req["filename"]
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


model = get_model(MODEL_NAME)


# @app.route("/", methods=["POST"])
# def index():
#     # serve static html file static/index.html
#     return render_template("static/index.html")


# @app.route("/predict", methods=["POST"])
# def predict():

#     req = request.get_json(force=True)
#     image = decode_request(req)
#     batch = preprocess(image)

#     predictions = model.predict(batch)

#     response = {
#         "classification": CLASSES[np.argmax(predictions[0])].upper(),
#         "confidence": f"{np.max(predictions[0])*100:.2f}%",
#         CLASSES[0]: f"{predictions[0][0]*100:.2f}%",
#         CLASSES[1]: f"{predictions[0][1]*100:.2f}%",
#         CLASSES[2]: f"{predictions[0][2]*100:.2f}%",
#     }

#     print("[+] results {}".format(response))

#     return jsonify(response)  # return it as json


# Convert from above flask implementation to FastAPI


HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "info"
RELOAD = True

app = FastAPI(
    title="Tumor Classification",
    description="Tumor Classification",
    version="0.1.0",
    swagger_ui_parameters={
        "docExpansion": "none",
        "syntaxHighlight.theme": "nord",
        "tryItOutEnabled": True,
        # "oauth2RedirectUrl": config.async_api.auth.oauth2_url,  # <-- Future?
    },
)


@app.get("/")
def index():
    return HTMLResponse(
        status_code=200,
        content=open(
            os.path.join(os.getcwd(), "api", "static", "index.html")
        ).read(),
    )


# upload image to the server
@app.post("/files/")
def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post("/predict/")
def predict(req: Request):
    decoded = decode_request(req.json)
    batch = preprocess(decoded)

    predictions = model.predict(batch)

    response = {
        "classification": CLASSES[np.argmax(predictions[0])].upper(),
        "confidence": f"{np.max(predictions[0])*100:.2f}%",
        CLASSES[0]: f"{predictions[0][0]*100:.2f}%",
        CLASSES[1]: f"{predictions[0][1]*100:.2f}%",
        CLASSES[2]: f"{predictions[0][2]*100:.2f}%",
    }

    print("[+] results {}".format(response))

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        reload=RELOAD,
    )
