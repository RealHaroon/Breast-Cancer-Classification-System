# Breast-Cancer-Classifier


# Breast Cancer Classifier (BUSI) — Benign vs Malignant (Deep Learning)

Breast ultrasound image classifier trained on the **BUSI** dataset (benign vs malignant).  
Pipeline includes: dataset preparation (mask removal + split), transfer learning, fine-tuning, threshold tuning, and deployment via FastAPI + a simple web UI.

> Disclaimer: Educational project only. Not medical advice.

***
![Breast Cancer Classifier](Breast Cancer Classifier.png)
## Project structure

```
breast-cancer-api/
  app.py
  models/
    best_model_finetuned.keras
  templates/
    index.html
  requirements.txt
```

***

## Tech stack

### Training (Kaggle)
- Python, TensorFlow / Keras
- Transfer learning with EfficientNetB0
- Metrics: Accuracy, AUC, Precision, Recall
- Callbacks: EarlyStopping + ModelCheckpoint

### Deployment (Local)
- FastAPI + Uvicorn
- Jinja2 template (HTML UI)
- JavaScript fetch + FormData (no page reload)
- Pillow (PIL) for image decoding/resizing

FastAPI file upload uses `UploadFile = File(...)`. [fastapi.tiangolo](https://fastapi.tiangolo.com/tutorial/request-files/)
Keras fine-tuning requires freezing/unfreezing layers and recompiling. [keras](https://keras.io/guides/transfer_learning/)
Dataset loading uses `image_dataset_from_directory(...)` which expects “folder-per-class”. [tensorflow.rstudio](https://tensorflow.rstudio.com/reference/keras/image_dataset_from_directory)

***

## Dataset

- Dataset: **BUSI (Breast Ultrasound Images Dataset)**  
- Classes: `benign`, `malignant` (we ignore `normal` for this v1)
- Notes:
  - BUSI often includes segmentation masks (files containing `mask` in the filename).  
  - For classification, we **exclude** mask images and train only on the raw ultrasound images.

***

## Training workflow (Kaggle)

### Step 1 — Prepare train/val/test splits (70/15/15)

We created a “processed” dataset folder for Keras:

```
/kaggle/working/busi_processed/
  train/
    benign/
    malignant/
  val/
    benign/
    malignant/
  test/
    benign/
    malignant/
splits.csv
```

Why:
- `/kaggle/input` is read-only (dataset mount)
- `/kaggle/working` is writable and becomes notebook output artifacts

We also saved `splits.csv` for reproducibility.

***

### Step 2 — Load datasets in TensorFlow

We used:

- `tf.keras.utils.image_dataset_from_directory(...)` [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory)
- `label_mode="binary"` for benign vs malignant [tensorflow.rstudio](https://tensorflow.rstudio.com/reference/keras/image_dataset_from_directory)
- `image_size=(224,224)` for EfficientNet input

Example config:

- Train dataset: shuffle = True
- Val/test: shuffle = False
- Performance: `cache().prefetch(tf.data.AUTOTUNE)` for faster input pipeline

`image_dataset_from_directory` automatically infers labels from folder names. [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory)

***

### Step 3 — Build model (Transfer Learning)

Model:
- Backbone: `tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet")`
- Head: GlobalAveragePooling → Dropout → Dense(1, sigmoid)

Stage 1 (transfer learning):
- Backbone frozen (`base.trainable = False`)
- Train only the final classifier head first

This is the standard Keras approach for transfer learning. [tensorflow](https://www.tensorflow.org/guide/keras/transfer_learning)

***

### Step 4 — Train (head training)

We trained with:

- Loss: `binary_crossentropy`
- Optimizer: Adam
- Metrics: Accuracy, AUC, Precision, Recall

Callbacks:
- `EarlyStopping(monitor="val_auc", mode="max", restore_best_weights=True)`
- `ModelCheckpoint(save_best_only=True, monitor="val_auc")` [keras](https://keras.io/api/callbacks/model_checkpoint/)
  
`ModelCheckpoint(save_best_only=True)` saves only the best model weights/file based on the monitored metric. [keras](https://keras.io/api/callbacks/model_checkpoint/)

Saved baseline model:
- `/kaggle/working/best_model.keras`

***

### Step 5 — Evaluate baseline on test set

We evaluated using `model.evaluate(test_ds, return_dict=True)` which is the standard evaluation method in Keras. [keras](https://keras.io/api/models/model_training_apis/)

We verified:
- The saved model and in-memory model matched (same metrics).

***

### Step 6 — Fine-tuning (unfreeze and train with low LR)

Goal: improve performance by letting some pretrained layers adapt to BUSI ultrasound images.

What we did:
1. Unfroze the backbone: `base.trainable = True`
2. Froze early layers and fine-tuned only deeper layers (`fine_tune_at = 200`)
3. Recompiled the model after changing `trainable` (important) [tensorflow](https://www.tensorflow.org/guide/keras/transfer_learning)
4. Used a small learning rate (e.g., `1e-5`)

Saved fine-tuned best model:
- `/kaggle/working/best_model_finetuned.keras`

Fine-tuning workflow is documented in the Keras transfer learning guide. [keras](https://keras.io/guides/transfer_learning/)

***

### Step 7 — Threshold tuning (decision threshold != 0.5)

The model outputs a probability for malignant (sigmoid output).  
Default classification uses threshold 0.5, but for medical screening it can be better to lower threshold to increase recall.

Process:
1. Predicted probabilities on validation set: `model.predict(...)` [apxml](https://apxml.com/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/making-predictions-predict)
2. Scanned thresholds from 0.05 to 0.95
3. Selected best threshold that satisfies:
   - **Recall ≥ 0.90** (high sensitivity)
   - Among those, highest precision

Final threshold:
- `THRESHOLD = 0.30`

This threshold significantly improved malignant recall on test set.

***

## Final model selection

We compared:
- Baseline model (`best_model.keras`)
- Fine-tuned model (`best_model_finetuned.keras`)

Final choice:
- **best_model_finetuned.keras**
- Threshold: **0.30** (probability of malignant)

Reason:
- Higher AUC and higher recall (better malignant detection) on test set.

***

## Exporting & downloading the model from Kaggle

The model files were saved in `/kaggle/working/`.  
Kaggle exposes this folder as notebook output artifacts (downloadable after saving/committing).

Final file downloaded locally:
- `models/best_model_finetuned.keras`

***

## Local deployment (FastAPI)

### API endpoints
- `GET /` → HTML upload UI
- `POST /predict` → accepts image upload and returns JSON prediction

FastAPI handles file uploads using `UploadFile = File(...)`. [fastapi.tiangolo](https://fastapi.tiangolo.com/reference/uploadfile/)

### Prediction logic
- Read uploaded image bytes
- Convert to RGB and resize to `(224,224)` using Pillow
- Convert to NumPy float array
- `prob = model.predict(...)`
- If `prob >= 0.30` → malignant, else benign

`model.predict()` is used to generate inference outputs in Keras. [apxml](https://apxml.com/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/making-predictions-predict)

***

## Running locally

### 1) Install dependencies
Create a virtual environment and install:

```bash
pip install -r requirements.txt
```

### 2) Start server
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open:
- http://127.0.0.1:8000 (UI)
- http://127.0.0.1:8000/docs (Swagger UI)

***

## How to test

### From UI
- Upload any BUSI benign/malignant ultrasound image.
- UI displays:
  - probability of malignant
  - threshold
  - final predicted label

### From terminal (optional)
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.png"
```

***

## Notes / limitations
- Dataset is small and imbalanced; results depend on the split.
- No patient-wise split metadata is used here (BUSI doesn’t always provide patient IDs in the same way as some other medical datasets).
- This is a demo classifier; real clinical systems require stronger validation, careful dataset governance, and clinical review.

***

## Next improvements (v2 ideas)
- Cross-validation for more stable evaluation
- Better augmentation (contrast/brightness for ultrasound)
- Calibration / better threshold selection strategy
- Use the BUSI masks for ROI cropping or segmentation-assisted classification
- Add Grad-CAM explanations

***

