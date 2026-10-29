from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = FastAPI(title="PPE Safety Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "running", "service": "PPE Safety Detection API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    results = model.predict(frame, conf=0.35, verbose=False)
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detections.append({
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": round(float(box.conf[0]), 4),
            "box": [round(float(x), 2) for x in box.xyxy[0].tolist()]
        })

    return {"detections": detections, "count": len(detections)}
