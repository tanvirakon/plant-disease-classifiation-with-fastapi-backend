# bckend
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
import torch.nn.functional as F

app = FastAPI()


# the frontend addresses allowed to access the FastAPI backend
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,  # enables cross-origin requests.
    allow_origins=origins,  # only the listed frontend addresses may connect.
    allow_credentials=True,  # allows cookies or authentication information.
    allow_methods=["*"],  # allows all HTTP methods, such as GET and POST.
    allow_headers=["*"],  # allows all request headers
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
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    result = {
        "class": CLASS_NAMES[predicted_idx.item()],
        "confidence": float(confidence.item()),
    }
    # print(result)
    return result  # <-- you also need a return statement


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)  # backend
