# Plant Disease Classifier API

FastAPI service for classifying plant leaf images with a trained PyTorch model.

## What is included

- `run.py` - API entry point with `/ping` and `/predict`
- `model.py` - CNN model definition
- `best_model.pth` - trained model weights
- `class_names.json` - class label mapping used by inference
- `plantvill/` - dataset organized by class folders
- `plant-project-codebasics.ipynb` - training notebook

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- FastAPI
- uvicorn
- pillow
- numpy

## Run locally

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Start the API:

```bash
uvicorn run:app --host localhost --port 8000
```

Or run the script directly:

```bash
python run.py
```

## Endpoints

- `GET /ping` - health check
- `POST /predict` - upload an image file and receive the predicted class and confidence

## Notes

- The model expects RGB images.
- Inputs are resized to 224x224 and normalized before inference.
- The repository ignores local tooling folders such as `.claude`, `.venv`, `__pycache__`, and notebook checkpoints.