import logging

from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import RegressionPreset, DataDriftPreset

logger = logging.getLogger(__name__)


def _create_data_definition(num_features, cat_features, target):
    definition = DataDefinition(
        numerical_columns=num_features + [target, "prediction"],
        categorical_columns=cat_features,
        regression=[
            Regression(
                target=target,
                prediction="prediction",
            )
        ],
    )
    return definition


def _extract_regression_metrics(report_dict):
    metrics = report_dict.get("metrics", [])

    def get_metric(metric_type, use_mean=False):
        metric = next(
            (m for m in metrics if m.get("config", {}).get("type") == metric_type),
            None,
        )

        if metric is None:
            return None

        value = metric.get("value")

        if use_mean and isinstance(value, dict):
            return value.get("mean")

        return value

    rmse = get_metric("evidently:metric_v2:RMSE")
    mae = get_metric("evidently:metric_v2:MAE", use_mean=True)
    r2 = get_metric("evidently:metric_v2:R2Score")
    mape = get_metric("evidently:metric_v2:MAPE", use_mean=True)

    return rmse, mae, r2, mape


def _extract_drift(report_dict):
    metrics = report_dict.get("metrics", [])

    drift_metric = next(
        (
            m
            for m in metrics
            if m.get("config", {}).get("type")
            == "evidently:metric_v2:DriftedColumnsCount"
        ),
        None,
    )

    if drift_metric is None:
        return False, 0

    count = int(drift_metric["value"]["count"])
    share = drift_metric["value"]["share"]
    threshold = drift_metric["config"]["drift_share"]

    drift_detected = share >= threshold

    return drift_detected, count


def generate_validation_report(
    reference_data, current_data, num_features, cat_features, target
):
    """
    In our case reference_data corresponds to the historical data (January)
    and current_data corresponds to the new data to evaluate (one week of February)
    """

    # Define the data definition (modern version of column mapping)
    # Ensure cnt is treated as a numerical column for drift calculation
    definition = _create_data_definition(num_features, cat_features, target)

    # Wrap pandas DataFrames into Dataset objects
    reference_dataset = Dataset.from_pandas(reference_data, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_data, data_definition=definition)

    # Initialize the Evidently report
    report = Report(metrics=[RegressionPreset(), DataDriftPreset()])

    # Run the report and convert it to a dictionary
    report_dict = report.run(
        reference_data=reference_dataset, current_data=current_dataset
    ).dict()

    print(report_dict)

    # extract regression metrics
    rmse, mae, r2, mape = _extract_regression_metrics(report_dict)

    # extract drift information
    drift_detected, drifted_columns_count = _extract_drift(report_dict)

    return rmse, mae, r2, mape, drift_detected, drifted_columns_count
