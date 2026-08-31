import datetime
import logging
from contextlib import asynccontextmanager
from typing import Any
import pandas as pd
import time

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
COLS_TO_DROP = ["yr", "cnt", "casual", "registered"]
PREDICTION = "prediction"
NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]

FEATURES_ORDERED = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


# --- Custom Exceptions ---
class EvaluationError(Exception):
    """Custom exception for errors during model evaluation."""


# --- FastAPI App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Starting API...")

    _train_and_predict_reference_model(TARGET, COLS_TO_DROP, NUM_FEATS, CAT_FEATS)

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
registry = CollectorRegistry()

# Counter 'api_requests_total', label par endpoint, method, et status code
api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    labelnames=["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "Duration of API requests in seconds",
    labelnames=["endpoint", "method", "status_code"],
    registry=registry,
)

model_rmse_score = Gauge(
    "model_rmse_score",
    "RMSE score of the trained model",
    registry=registry,
)

model_mae_score = Gauge(
    "model_mae_score",
    "MAE score of the trained model",
    registry=registry,
)

model_r2_score = Gauge(
    "model_r2_score",
    "R2 score of the trained model",
    registry=registry,
)

evidently_data_drift_detected_status = Gauge(
    "evidently_data_drift_detected_status",
    "Status of data drift detected by Evidently",
    registry=registry,
)


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
    # dteday: datetime.date = Field(
    #     ...,
    #     example="2011-01-01",
    #     description="Date of the record in YYYY-MM-DD format.",
    # )


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

    logger.info("Starting prediction request")
    # Values to track for logging and metrics
    start_time = time.time()  # Début du timer pour la durée de la requête
    status_code = "200"
    try:
        # Convert input data to a DataFrame
        logger.info(f"Received input data: {input_data}")

        # validate input data
        input_data_dict = input_data.model_dump()
        for feature in FEATURES_ORDERED:
            if feature not in input_data_dict:
                logger.error(f"Missing required feature: {feature}")
                status_code = "400"
                raise HTTPException(
                    status_code=400, detail=f"Missing required feature: {feature}"
                )

        input_df = pd.DataFrame([input_data.model_dump()])

        X = input_df[FEATURES_ORDERED]

        # Ensure the model is loaded
        logger.info("Checking if the model is loaded")
        if not hasattr(app.state, "model"):
            logger.error("Model is not loaded")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model is not loaded")

        # Make prediction
        logger.info("Making prediction with the loaded model")
        prediction = app.state.model.predict(X)
        if not prediction:
            logger.error("Prediction failed, no result returned")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Prediction failed")

        logger.info(f"Prediction result: {prediction[0]}")

        return PredictionOutput(predicted_count=prediction[0])
    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except EvaluationError as e:
        logger.error(
            f"Error during prediction for input data: {input_data_dict}... Error: {e}"
        )
        status_code = "500"
        raise HTTPException(
            status_code=500, detail=f"Prediction failed due to an internal error: {e}"
        )
    finally:
        end_time = time.time()
        # Durée de la requête
        duration = end_time - start_time
        api_request_duration_seconds.labels(
            endpoint="/predict", method="POST", status_code=status_code
        ).observe(duration)
        api_requests_total.labels(
            endpoint="/predict", method="POST", status_code=status_code
        ).inc()
