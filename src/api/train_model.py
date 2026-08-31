import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _fetch_data(path="data/hour.csv"):
    logger.info(f"Fetching data from {path}")
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Data fetched successfully from {path}")
    return df


def _process_data(df, num_feats, cat_feats):
    df["dteday"] = pd.to_datetime(df["dteday"])

    date_start = "2011-01-01"
    date_end = "2011-12-31"

    jan_data = df[(df["dteday"] >= date_start) & (df["dteday"] <= date_end)]

    df = jan_data.copy()

    logger.info("Processing numerical and categorical features")
    for col in num_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Processed numerical feature: {col}")
        # replace NaN values with mean
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_feats:
        df[col] = df[col].astype("category")
        logger.info(f"Processed categorical feature: {col}")
        # replace NaN values with the most frequent value for categorical features
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def _save_model(model, model_path):
    # Create 'models' directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    logger.info(f"Saving model to {model_path}")
    with open(model_path, "wb") as f_out:
        joblib.dump(model, f_out)
    logger.info("Model saved successfully")


def _train_and_predict_reference_model(target, all_targets, num_feats, cat_feats):

    df = _fetch_data()

    jan_df = _process_data(df, num_feats, cat_feats)

    # Split the data into features and target variable
    X = jan_df.drop(columns=all_targets)
    y = jan_df[target]

    if not X.index.equals(y.index):
        logger.error("X et y n'ont pas les mêmes index avant le filtrage")
        raise ValueError("X et y n'ont pas les mêmes index avant le filtrage")

    X_jan = X.copy()
    y_period = y.copy()

    X_jan = X_jan.drop(columns=["dteday"])

    X_jan = X_jan.reset_index(drop=True)
    y_period = y_period.reset_index(drop=True)

    # Initialize and train a Random Forest Regressor model
    logger.info("Training Random Forest Regressor model")
    model = RandomForestRegressor()
    model.fit(X_jan, y_period)
    logger.info("Random Forest Regressor model trained successfully")

    # Save the trained model
    logger.info("Saving the trained model")
    _save_model(model, "./models/bike_share_reference_model.bin")
    logger.info("Trained model saved successfully")
