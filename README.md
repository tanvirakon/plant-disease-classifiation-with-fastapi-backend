# Plant Disease Classifier

FastAPI backend + Streamlit frontend for classifying plant leaf images with a trained PyTorch CNN (38 PlantVillage classes).

## What is included

- `run.py` - FastAPI backend entry point with `/ping` and `/predict`
- `app.py` - Streamlit frontend (upload leaf image, view prediction + confidence)
- `model.py` - `PlantVillageCNN` model definition
- `best_model.pth` - trained model weights
- `class_names.json` - class label mapping used by inference
- `plant-project-codebasics.ipynb` - training notebook (local `plantvill/` dataset, commented transforms)

## Requirements

- Python 3.10+
- torch
- torchvision
- FastAPI
- uvicorn
- pillow
- numpy
- streamlit
- requests

Install:

```bash
pip install torch torchvision fastapi uvicorn pillow numpy streamlit requests
```

## Run locally

1. Start the backend (port 8000):

```bash
uvicorn run:app --reload
```

Or:

```bash
python run.py
```

2. Start the frontend (new terminal):

```bash
streamlit run app.py
```

The frontend posts uploads to `http://localhost:8000/predict`.

## Endpoints

- `GET localhost:8000/ping` - health check
- `POST localhost:8000/predict` - upload an image file (`file` field), receive `{class, confidence}`
- `GET localhost:8000/docs` - interactive Swagger UI
- Frontend via Streamlit shows predicted label, confidence metric, and progress bar.

## Notes

- The model expects RGB images.
- Inputs are resized to 224x224 and normalized before inference.
- The repository ignores local tooling folders such as `.claude`, `.venv`, `__pycache__`, and notebook checkpoints.