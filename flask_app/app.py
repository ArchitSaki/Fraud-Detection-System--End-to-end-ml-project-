from flask import Flask, render_template, request
import mlflow
import pickle
import pandas as pd
import time
import dagshub
import os
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

import warnings
warnings.filterwarnings("ignore")

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ArchitSaki"
repo_name = "Fraud-Detection-System--End-to-end-ml-project-"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# # MLflow setup
# mlflow.set_tracking_uri('https://dagshub.com/ArchitSaki/Fraud-Detection-System--End-to-end-ml-project-.mlflow')
# dagshub.init(repo_owner='ArchitSaki', repo_name='Fraud-Detection-System--End-to-end-ml-project-', mlflow=True)

# Load models
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

app = Flask(__name__)

# Prometheus metrics
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry
)

# Preprocessing function
def preprocess_input(data):

    df = pd.DataFrame([data])

    type_mapping = {
        "PAYMENT":0,
        "TRANSFER":1,
        "CASH_OUT":2,
        "DEBIT":3,
        "CASH_IN":4
    }

    # df["type"] = df["type"].map(type_mapping)
    df["type"] = label_encoder.transform(df["type"].astype(str))

    if df["type"].isnull().any():
        raise ValueError("Invalid transaction type")
    # Feature Engineering
    df["orgBalanceDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["destBalanceDiff"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df.drop(["nameOrig","nameDest","step"], axis=1, inplace=True, errors="ignore")

    df = df[
        [
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "orgBalanceDiff",
            "destBalanceDiff"
        ]
    ]

    # 🔹 Align features with scaler
    expected_features = list(scaler.feature_names_in_)

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    df_scaled = scaler.transform(df)

    return df_scaled


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    response = render_template("index.html", result=None)

    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    data = {
        "type": request.form["type"],
        "amount": float(request.form["amount"]),
        "nameOrig": request.form["nameOrig"],
        "oldbalanceOrg": float(request.form["oldbalanceOrg"]),
        "newbalanceOrig": float(request.form["newbalanceOrig"]),
        "nameDest": request.form["nameDest"],
        "oldbalanceDest": float(request.form["oldbalanceDest"]),
        "newbalanceDest": float(request.form["newbalanceDest"]),
    }

    processed_data = preprocess_input(data)

    # Get probability
    proba = model.predict_proba(processed_data)[0][1]

    # Apply threshold
    if proba > 0.2:
        prediction = "Fraud Transaction"
        pred_class = 1
    else:
        prediction = "Legitimate Transaction"
        pred_class = 0

    # Prometheus metric
    PREDICTION_COUNT.labels(prediction=str(pred_class)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction, probability=round(proba, 4))


@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)