from cgitb import html
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


# upload image file and display the uploaded image after the upload
@app.get("/")
async def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Upload File</title>
    </head>
    <body>
    <h1>Upload File</h1>
    <input id="upload" type="file" />
    <div id="image" width="300" height="300"></div>
    <script>
    const upload = document.getElementById('upload');
    const image = document.getElementById('image');
    upload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
            image.innerHTML = '<img src="' + e.target.result + '" width="300" height="300">';
        };
        reader.readAsDataURL(file);
    });
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
    # return HTMLResponse(html)


# upload file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


# get file
@app.get("/file/{filename}")
async def get_file(filename: str):
    return {"filename": filename}
