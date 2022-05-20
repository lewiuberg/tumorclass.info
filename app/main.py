import os
import shutil
from pathlib import Path

import uvicorn
from confprint import prefix_printer
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modules.model import predicts
from pyconfs import Configuration
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Functional, Sequential


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_PATH = f"{PROJECT_ROOT}/models"

cfg = Configuration.from_file(f"{PROJECT_ROOT}/pyproject.toml")
CLASSES = cfg.env.classes
WIDTH_HEIGHT = (cfg.env.width, cfg.env.height)

cnn_custom: Sequential = load_model(f"{MODELS_PATH}/cnn_custom_model.h5")
cnn_vgg16: Functional = load_model(f"{MODELS_PATH}/cnn_vgg16_model.h5")
cnn_mobilenet: Sequential = load_model(f"{MODELS_PATH}/cnn_mobilenet_model.h5")
cnn_densenet121: Functional = load_model(
    f"{MODELS_PATH}/cnn_densenet121_model.h5"
)
cnn_alexnet: Functional = load_model(f"{MODELS_PATH}/cnn_alexnet_model.h5")

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

origins = ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """
    Get the home page.

    Parameters
    ----------
    request : Request
        The request object.

    Returns
    -------
    HTMLResponse
        The home page.
    """
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def get_about(request: Request):
    """
    Get the about page.

    Parameters
    ----------
    request : Request
        The request object.

    Returns
    -------
    HTMLResponse
        The about page.
    """
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/data", response_class=HTMLResponse)
def get_data(request: Request):
    """
    Get the data page.

    Parameters
    ----------
    request : Request
        The request object.

    Returns
    -------
    HTMLResponse
        The data page.
    """
    return templates.TemplateResponse("data.html", {"request": request})


@app.get("/literature-review", response_class=HTMLResponse)
def get_literature_review(request: Request):
    """
    Get the literature review page.

    Parameters
    ----------
    request : Request
        The request object.

    Returns
    -------
    HTMLResponse
        The literature review page.
    """
    return Response(
        content=open(
            f"{PROJECT_ROOT}/static/html/literature-review.html", "rb"
        ).read(),
        media_type="text/html",
    )


@app.get("/conference-paper", response_class=HTMLResponse)
def get_conference_paper(request: Request):
    """
    Get the conference paper page.

    Parameters
    ----------
    request : Request
        The request object.

    Returns
    -------
    HTMLResponse
        The conference paper page.
    """
    return templates.TemplateResponse(
        "conference-paper.html", {"request": request}
    )


@app.post("/upload")
def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Upload a file.

    Parameters
    ----------
    request : Request
        The request object.
    file : UploadFile, optional
        The file to upload.

    Returns
    -------
    HTMLResponse
        The home page.
    """
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


@app.get("/delete")
def delete_file(request: Request):
    """
    Delete the uploaded file.

    Parameters
    ----------
    request : Request
        The request object.

    Returns
    -------
    HTMLResponse
        The home page.
    """
    p_information("Deleting file...")
    # get the file name from the request
    filename = request.query_params["filename"]
    # use static mount to delete the file from the temp folder
    file_path = str(project_root / "static" / "temp" / filename)
    # file_path = Path(project_root / "temp" / filename)
    os.remove(file_path)
    return templates.TemplateResponse(
        "home.html", {"request": request, "filename": None}
    )


@app.post("/predict")
def prediction(
    request: Request,
    filename: str = Form(...),
    custom_model: bool = Form(False),
    vgg16_model: bool = Form(False),
    mobilenet_model: bool = Form(False),
    densenet121_model: bool = Form(False),
    alexnet_model: bool = Form(False),
):
    """
    Predict the tumor type.

    Parameters
    ----------
    request : Request
        The request object.
    filename : str, optional
        The file name.
    custom_model : bool, optional
        Whether to use the custom model.
    vgg16_model : bool, optional
        Whether to use the vgg16 model.
    mobilenet_model : bool, optional
        Whether to use the mobilenet model.
    densenet121_model : bool, optional
        Whether to use the densenet121 model.
    alexnet_model : bool, optional
        Whether to use the alexnet model.

    Returns
    -------
    HTMLResponse
        The home page.
    """
    selected_models = {}
    if vgg16_model:
        selected_models["cnn_vgg16"] = cnn_vgg16
    if mobilenet_model:
        selected_models["cnn_mobilenet"] = cnn_mobilenet
    if custom_model:
        selected_models["cnn_custom"] = cnn_custom
    if alexnet_model:
        selected_models["cnn_alexnet"] = cnn_alexnet
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
            "alexnet_model": alexnet_model,
        },
    )


@app.on_event("startup")
async def startup():
    """
    Startup event.

    Makes a temp folder to store the uploaded files.
    """
    p_information("Starting app...")
    if not os.path.exists(project_root / "static" / "temp"):
        os.mkdir(project_root / "static" / "temp")


@app.on_event("shutdown")
async def shutdown():
    """
    Shutdown event.

    Deletes the temp folder.
    """
    p_information("Shutting down...")
    shutil.rmtree(project_root / "static" / "temp")


if __name__ == "__main__":
    # Not used when running VSCode Run and Debug
    print("Running without VSCode Run and Debug")

    launch = Configuration.from_file(f"{PROJECT_ROOT}/.vscode/launch.json")
    uvicorn.run(
        "main:app",
        host=launch["configurations"][1]["args"][3],
        port=int(launch["configurations"][1]["args"][5]),
        log_level="info",
        reload=True,
    )
