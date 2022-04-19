from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


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


# upload file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


# get file
@app.get("/file/{filename}")
async def get_file(filename: str):
    return {"filename": filename}
