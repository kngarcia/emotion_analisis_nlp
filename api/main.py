from fastapi import FastAPI
from schemas import PredictionRequest
from inference import predict, get_model_info
import pandas as pd
import time
import matplotlib.pyplot as plt

app = FastAPI(
    title="Sentiment140 Inference API",
    version="1.0"
)

@app.get("/model_info")
def model_info():
    return get_model_info()

@app.post("/predict")
def predict_sentiment(request: PredictionRequest):

    predictions = predict(request.texts)

    return {
        "predictions": predictions
    }

@app.get("/ablation_summary")
def ablation_summary():

    df = pd.read_csv("../reports/ablation_results.csv")

    table = df.to_dict(orient="records")

    best = df.loc[df["f1_score"].idxmax()]

    conclusion = f"""
    The best configuration was {best['model']}
    with encoding {best['encoding']}
    achieving F1 score {best['f1_score']}.
    """

    return {
        "table": table,
        "conclusion": conclusion
    }


@app.get("/comparison")
def comparison():

    results = {
        "classical_model": {
            "f1_score": 0.86,
            "runtime_train": "45 sec",
            "runtime_test": "2 sec"
        },
        "distilbert": {
            "f1_score": 0.89,
            "runtime_train": "0 sec (pretrained)",
            "runtime_test": "35 sec"
        }
    }

    return results

@app.get("/work_distribution")
def work_distribution():

    table = [
        {"member": "Nicolas", "experiments": "Baseline + Neural Network"},
        {"member": "Member2", "experiments": "Preprocessing"},
        {"member": "Member3", "experiments": "Tree Models"}
    ]

    return {"distribution": table}