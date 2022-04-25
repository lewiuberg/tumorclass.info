import os
import shutil
from pathlib import Path

from confprint import prefix_printer
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.model import predicts
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Functional, Sequential


# import numpy as np
# import uvicorn


PROJECT_ROOT = Path(__file__).parent.parent

CLASSES = ["normal", "lgg", "hgg"]
MODELS_PATH = f"{PROJECT_ROOT}/models"
WIDTH_HEIGHT = (224, 224)

cnn_custom: Sequential = load_model(f"{MODELS_PATH}/cnn_custom_model.h5")
cnn_vgg16: Functional = load_model(f"{MODELS_PATH}/cnn_vgg16_model.h5")
cnn_mobilenet: Sequential = load_model(f"{MODELS_PATH}/cnn_mobilenet_model.h5")
cnn_densenet121: Functional = load_model(
    f"{MODELS_PATH}/cnn_densenet121_model.h5"
)

p_information = prefix_printer(
    "INFO",
    prefix_color="blue",
    frame_left="",
    frame_right="",
    whitespace=4,
)

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
    "/static",
    StaticFiles(directory=str(project_root / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(project_root / "templates"))


# get index page
@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def get_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/data", response_class=HTMLResponse)
def get_data(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})


@app.get("/literature-review", response_class=HTMLResponse)
def get_literature_review(request: Request):
    # return StaticFiles
    return Response(
        content=open(
            f"{PROJECT_ROOT}/static/html/literature-review.html", "rb"
        ).read(),
        media_type="text/html",
    )


@app.get("/conference-paper", response_class=HTMLResponse)
def get_conference_paper(request: Request):
    return "/static/html/literature-review.html"


@app.post("/upload")
def upload_file(request: Request, file: UploadFile = File(...)):
    p_information("Uploading file...")
    # get the file.filename extension and rename it to f'image.{extension}
    # file.filename = f"image.{file.filename.split('.')[-1]}"
    filename = file.filename
    # use static mount to store the file in the temp folder
    file_path = str(project_root / "static" / "temp" / filename)
    # file_path = Path(project_root / "temp" / filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return templates.TemplateResponse(
        "home.html", {"request": request, "filename": filename}
    )


@app.post("/predict")
def prediction(
    request: Request,
    filename: str = Form(...),
    custom_model: bool = Form(False),
    vgg16_model: bool = Form(False),
    mobilenet_model: bool = Form(False),
    densenet121_model: bool = Form(False),
):
    selected_models = {}
    if custom_model:
        selected_models["cnn_custom"] = cnn_custom
    if vgg16_model:
        selected_models["cnn_vgg16"] = cnn_vgg16
    if mobilenet_model:
        selected_models["cnn_mobilenet"] = cnn_mobilenet
    if densenet121_model:
        selected_models["cnn_densenet121"] = cnn_densenet121
    # get filename from the request
    predictions = predicts(
        path=f"{PROJECT_ROOT}/static/temp/{filename}",
        target_size=WIDTH_HEIGHT,
        models=selected_models,
        classes=CLASSES,
    )
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "filename": filename,
            "predictions": predictions,
            "classes": CLASSES,
            "custom_model": custom_model,
            "vgg16_model": vgg16_model,
            "mobilenet_model": mobilenet_model,
            "densenet121_model": densenet121_model,
        },
    )


@app.on_event("startup")
async def startup():
    p_information("Starting app...")
    if not os.path.exists(project_root / "static" / "temp"):
        os.mkdir(project_root / "static" / "temp")


@app.on_event("shutdown")
async def shutdown():
    p_information("Shutting down...")
    shutil.rmtree(project_root / "static" / "temp")


# TODO: Submit 2 supervisor reports, one for today, and one for Friday.
# TODO: Make a Flowchart of the custom CNN.
# TODO: Make diagram of architecture(s).
# TODO: Finalise the "webpage" <- Printscreens
# TODO: Expand on the Lit. Rev. by looking at the guidelines to make the thesis
# TODO: Look for more testing data.
