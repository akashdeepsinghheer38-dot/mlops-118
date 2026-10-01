
import pandas as pd
import boto3
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
import joblib


# ============================================================
# S3 CONFIGURATION
# ============================================================

BUCKET = "mlops-house-prediction"

KEY = "proccessed/2026-09-25/Mlops_house_predication_clean_v1.csv"


# ============================================================
# FETCH DATA FROM S3
# ============================================================

s3 = boto3.client("s3")


def fetch_data():

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=KEY
    )

    df = pd.read_csv(
        StringIO(
            obj["Body"].read().decode("utf-8")
        )
    )

    return df


df = fetch_data()

print(f"Fetched shape: {df.shape}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]

X = df[FEATURES]

y = df["price"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# MODEL
# ============================================================

n_estimators = 150

max_depth = 8


model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

print("Training model...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

preds = model.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        preds
    )
)

r2 = r2_score(
    y_test,
    preds
)


print("\nModel Evaluation")
print("-" * 40)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = "model.joblib"

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print(f"Model file: {MODEL_PATH}")
