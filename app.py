from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

MODEL_PATH = "models/final_efficientnetb0_3class.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["normal", "benign", "malignant"]  # 0/1/2

# Force-malignant threshold (tune this; lower => more malignant predictions)
FORCE_MALIGNANT_T = 0.30

# Optional uncertainty flags (separate from forcing)
UNCERTAINTY_MARGIN = 0.05
UNCERTAINTY_TOP1_MAX = 0.60

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)        # [0..255]
    arr = np.expand_dims(arr, axis=0)            # (1,224,224,3)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()  # UploadFile.read() [web:770]
    x = preprocess_image(contents)

    probs = model.predict(x, verbose=0)[0]  # (3,)
    probs = np.asarray(probs, dtype=np.float32)

    # argmax prediction
    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1])

    argmax_label = CLASS_NAMES[top1]
    p1 = float(probs[top1])
    p2 = float(probs[top2])
    margin = p1 - p2

    prob_malignant = float(probs[2])

    # FORCE malignant if prob >= threshold (policy change)
    forced = False
    final_label = argmax_label
    if prob_malignant >= FORCE_MALIGNANT_T:
        final_label = "malignant"
        forced = (argmax_label != "malignant")

    # Optional uncertainty indicator
    uncertain = (margin < UNCERTAINTY_MARGIN) and (p1 < UNCERTAINTY_TOP1_MAX)

    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}

    return JSONResponse(content={
        "filename": file.filename,
        "prediction": final_label,          # final decision
        "argmax_prediction": argmax_label,  # what model top-1 was
        "forced_malignant": forced,         # True if we overrode argmax
        "threshold": float(FORCE_MALIGNANT_T),

        "probs": probs_dict,
        "prob_malignant": prob_malignant,

        "top2": [
            {"label": CLASS_NAMES[top1], "prob": float(probs[top1])},
            {"label": CLASS_NAMES[top2], "prob": float(probs[top2])},
        ],
        "margin": float(margin),
        "uncertain": bool(uncertain),
    })
