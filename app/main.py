import base64
import io
import os
import shutil
from pathlib import Path

import numpy as np
from confprint import prefix_printer
from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


p_information = prefix_printer(
    "INFO",
    prefix_color="blue",
    frame_left="",
    frame_right="",
    whitespace=4,
)

CLASSES = ["normal", "lgg", "hgg"]
MODELS_PATH = (
    "/Users/lewiuberg/Documents/Data Science/Projects/tumorclass.info/models/"
)
MODEL_NAME = "cnn_densenet121"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

model = load_model(os.path.join(MODELS_PATH, f"{MODEL_NAME}_model.h5"))

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
    debug=True,
)

project_root = Path(__file__).parent.parent

app.mount(
    "/assets",
    StaticFiles(directory=str(project_root / "assets")),
    name="assets",
)

templates = Jinja2Templates(directory=str(project_root / "templates"))
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# upload image file to memory
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # Read image file as PIL image
    img = Image.open(io.BytesIO(file.file.read()))
    # If image is not RGB, convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize image to 224x224 and interpolate with nearest neighbour
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)
    # Convert image to array
    img_array = img_to_array(img)
    # Expand dimensions to create batch axis
    img_array = expand_dims(img_array, 0)
    # Predict
    predictions = model.predict(img_array)

    return {
        "classification": CLASSES[np.argmax(predictions[0])].upper(),
        "confidence": f"{np.max(predictions[0])*100:.2f}%",
        # # use the CLASSES
        CLASSES[0]: f"{predictions[0][0]*100:.2f}%",
        CLASSES[1]: f"{predictions[0][1]*100:.2f}%",
        CLASSES[2]: f"{predictions[0][2]*100:.2f}%",
    }


# use index.html as the homepage
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
