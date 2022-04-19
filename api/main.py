import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from tensorflow.keras.models import load_model


CLASSES = ["normal", "lgg", "hgg"]
MODELS_PATH = (
    "/Users/lewiuberg/Documents/Data Science/Projects/tumorclass.info/models/"
)
MODEL_NAME = "cnn_densenet121"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


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


app = FastAPI(
    swagger_ui_parameters={
        "docExpansion": "none",
        "syntaxHighlight.theme": "nord",
        "tryItOutEnabled": True,
    }
)


api_root = Path(__file__).parent


app.mount(
    "/static", StaticFiles(directory=str(api_root / "static")), name="static"
)

templates = Jinja2Templates(directory=str(api_root / "templates"))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
