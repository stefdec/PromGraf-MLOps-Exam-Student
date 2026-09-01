import datetime
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset, RegressionPreset
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_model import train_and_predict_reference_model
from reports import generate_validation_report

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
    drift_detected: int | None
    evaluated_items: int


# --- FastAPI App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Starting API...")

    X_jan, y_jan = train_and_predict_reference_model(
        TARGET, COLS_TO_DROP, NUM_FEATS, CAT_FEATS
    )

    # Save the reference data for later use in evaluation
    app.state.X_jan = X_jan
    app.state.y_jan = y_jan

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


# --- Middleware for logging request metrics ---
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    api_request_duration_seconds.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=str(response.status_code),
    ).observe(duration)

    api_requests_total.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=str(response.status_code),
    ).inc()

    return response


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

    try:
        logger.info(f"Received input data: {input_data}")

        # Convert input data to dict
        input_data_dict = input_data.model_dump()

        # Validate expected features
        for feature in FEATURES_ORDERED:
            if feature not in input_data_dict:
                logger.error(f"Missing required feature: {feature}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required feature: {feature}",
                )

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data_dict])
        X = input_df[FEATURES_ORDERED]

        # Ensure the model is loaded
        logger.info("Checking if the model is loaded")
        if not hasattr(app.state, "model"):
            logger.error("Model is not loaded")
            raise HTTPException(
                status_code=500,
                detail="Model is not loaded",
            )

        # Make prediction
        logger.info("Making prediction with the loaded model")
        prediction = app.state.model.predict(X)

        if prediction is None or len(prediction) == 0:
            logger.error("Prediction failed, no result returned")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed",
            )

        logger.info(f"Prediction result: {prediction[0]}")

        return PredictionOutput(predicted_count=prediction[0])

    except HTTPException:
        raise

    except (EvaluationError, KeyError, ValueError) as e:
        logger.error(
            f"Error during prediction for input data: {input_data_dict}. Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed due to an internal error: {e}",
        )


@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(input_data: EvaluationData):
    logger.info("Starting evaluation request")

    try:
        # Convert input data to a DataFrame
        logger.info("Received evaluation input data")

        input_data_dict = input_data.model_dump()
        evaluation_df = pd.DataFrame(input_data_dict["data"])

        # Ensure the model is loaded
        logger.info("Checking if the model is loaded for evaluation")
        if not hasattr(app.state, "model"):
            logger.error("Model is not loaded")
            raise HTTPException(status_code=500, detail="Model is not loaded")

        # Make predictions
        logger.info("Making predictions for evaluation on the received data")
        X_feb = evaluation_df[FEATURES_ORDERED]
        y_feb = evaluation_df["cnt"]
        y_feb_pred = app.state.model.predict(X_feb)

        # Preparing the reference data and current data for the validation report
        reference_data = app.state.X_jan.copy()
        reference_data["cnt"] = app.state.y_jan.copy()
        reference_data["prediction"] = app.state.model.predict(app.state.X_jan)

        current_data = X_feb.copy()
        current_data["cnt"] = y_feb.copy()
        current_data["prediction"] = y_feb_pred.copy()

        rmse, mae, r2, mape, drift_detected, drifted_columns_count = (
            generate_validation_report(
                reference_data=reference_data,
                current_data=current_data,
                num_features=NUM_FEATS,
                cat_features=CAT_FEATS,
                target="cnt",
            )
        )

        model_rmse_score.set(rmse)
        model_mae_score.set(mae)
        model_r2_score.set(r2)
        logger.info(
            f"Evaluation results - RMSE: {rmse}, MAE: {mae}, R2: {r2}, Drift Detected: {drift_detected}, Drifted Columns Count: {drifted_columns_count}"
        )

        if drift_detected:
            logger.warning(
                f"Data drift detected! Number of drifted columns: {drifted_columns_count}"
            )
            evidently_data_drift_detected_status.set(1)
        else:
            evidently_data_drift_detected_status.set(0)

        return EvaluationReportOutput(
            message="Evaluation completed successfully",
            rmse=rmse,
            mape=mape,
            mae=mae,
            r2score=r2,
            drift_detected=drift_detected,
            evaluated_items=len(evaluation_df),
        )
    except HTTPException:
        raise
    except (EvaluationError, KeyError, ValueError) as e:
        logger.error(f"Error during evaluation for input data... Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Evaluation failed due to an internal error: {e}"
        )


@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")
