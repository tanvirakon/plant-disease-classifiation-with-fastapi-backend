import io
import json
import torch
import uvicorn
import numpy as np
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model import PlantVillageCNN

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PlantVillageCNN(n_classes=38)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
model.to(device)

infer_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_batch = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return {
        "class": CLASS_NAMES[predicted_idx.item()],
        "confidence": float(confidence.item()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
