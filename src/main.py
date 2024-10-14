import os
import pickle

import pandas as pd
from fastapi import FastAPI, UploadFile, File

from utils import download_data_from_s3
from src.preprocessing import prepare_dataset

app = FastAPI(title="Credit Card Fraud Detection API")


@app.get("/")
def root():
    return (
        "API is up and running! Visit "
        "https://credit-fraud-detection-mlops.herokuapp.com/docs for documentation."
    )


@app.post("/predict/")
async def make_prediction(csv_file: UploadFile = File(...)):
    # Read the uploaded CSV file into a DataFrame
    dataframe = pd.read_csv(csv_file.file)
    processed_data = prepare_dataset(dataframe, training=False)

    X_test = processed_data["X_test"]
    y_test = processed_data["y_test"]

    model_filename = "trained_model"
    
    if not os.path.isfile(model_filename):
        download_data_from_s3(
            bucket_name="credit-fraud-mlops-artifacts",
            object_key="mtrained_model",
            download_path=model_filename
        )

    with open(model_filename, "rb") as model_file:
        classifier = pickle.load(model_file)

    print(f"Loaded model: {classifier}")

    accuracy = classifier.score(X_test, y_test)
    predictions = classifier.predict(X_test).tolist()

    return {
        "model_accuracy": accuracy,
        "predicted_labels": predictions
    }
