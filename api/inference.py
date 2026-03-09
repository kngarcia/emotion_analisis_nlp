import joblib
import json

MODEL_PATH = "../models/best_model.pkl"
CARD_PATH = "../models/model_card_LogReg_C1.0_lbfgs.json"

model = joblib.load(MODEL_PATH)

with open(CARD_PATH) as f:
    model_card = json.load(f)


def predict(texts):
    preds = model.predict(texts)
    return preds.tolist()


def get_model_info():
    return model_card