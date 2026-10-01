
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]


app = FastAPI(
    title="House Price Predictor",
    version="1.0"
)


# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Model loading failed: {e}")


class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=200)
    age_years: int = Field(..., gt=0, le=10)
    garage: int = Field(..., ge=0, le=10)
    location_score: int = Field(..., ge=1, le=10)


@app.get("/health")
def health():

    if model is None:
        return {
            "status": "unhealthy",
            "model": "not loaded"
        }

    return {
        "status": "healthy",
        "model": "loaded"
    }


@app.post("/predict")
def predict(features: HouseFeatures):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    try:
        input_df = pd.DataFrame(
            [features.model_dump()],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        return {
            "predicted_price": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Serve frontend
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )


@app.get("/")
def frontend():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "message": "House Price Predictor API is running",
            "docs": "/docs"
        }

    return FileResponse(index_file)

