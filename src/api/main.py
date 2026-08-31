import datetime
import logging
from contextlib import asynccontextmanager
from typing import Any
import pandas as pd

import joblib
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from train_model import _train_and_predict_reference_model

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables for Model and Data ---
TARGET = "cnt"
ALL_TARGETS = ["cnt", "casual", "registered"]
PREDICTION = "prediction"
NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]


# --- FastAPI App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Starting API...")

    _train_and_predict_reference_model(TARGET, ALL_TARGETS, NUM_FEATS, CAT_FEATS)

    # load the trained model for inference
    logger.info("Loading the trained model for inference")
    app.state.model = joblib.load("./models/bike_share_reference_model.bin")
    logger.info("Model loaded successfully -- Application ready")

    yield

    # --- Shutdown ---
    logger.info("Stopping application...")


app = FastAPI(
    lifespan=lifespan,
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0",
)

# --- Prometheus Metrics Definitions ---


# --- Data Ingestion and Preparation Functions ---


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(
        ...,
        example="2011-01-01",
        description="Date of the record in YYYY-MM-DD format.",
    )


class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)


class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description="List of data points, each containing features and the true target ('cnt').",
    )
    evaluation_period_name: str = Field(
        "unknown_period",
        description="Name of the period being evaluated (e.g., 'week1_february').",
    )
    model_config = {"arbitrary_types_allowed": True}


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: float | None
    mape: float | None
    mae: float | None
    r2score: float | None
    drift_detected: int
    evaluated_items: int


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput):
    # Convert input data to a DataFrame
    logger.info(f"Received input data: {input_data}")
    input_df = pd.DataFrame([input_data.model_dump()])

    # Ensure the model is loaded
    logger.info("Checking if the model is loaded")
    if not hasattr(app.state, "model"):
        logger.error("Model is not loaded")
        raise RuntimeError("Model is not loaded")

    # Make prediction
    logger.info("Making prediction with the loaded model")
    prediction = app.state.model.predict(input_df)
    logger.info(f"Prediction result: {prediction[0]}")

    return PredictionOutput(predicted_count=prediction[0])
