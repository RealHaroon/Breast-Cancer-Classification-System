from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

MODEL_PATH = "models/best_model_finetuned.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.30  # your tuned threshold

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)  # 0..255
    arr = np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)
    return arr

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # UploadFile/File is the standard FastAPI way to receive uploaded files [web:385][web:387]
    contents = await file.read()
    x = preprocess_image(contents)

    prob = float(model.predict(x, verbose=0)[0][0])  # sigmoid output
    label = "malignant" if prob >= THRESHOLD else "benign"

    return JSONResponse({
        "filename": file.filename,
        "prob_malignant": prob,
        "threshold": THRESHOLD,
        "prediction": label
    })
