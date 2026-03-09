import pickle
import json
import re
from pathlib import Path
from typing import List, Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/best_model.pkl"
MODEL_CARD_PATH = BASE_DIR / "models/model_card_LogReg_C1.0_lbfgs.json"

# ===============================
# Función de preprocesamiento estable
# ===============================
_punctuation_pattern = re.compile(r"[^\w\s]")
_elongation_pattern = re.compile(r"(.)\1{2,}")

def full_preprocessing(texts):
    processed = []
    for text in texts:
        if text is None:
            text = ""
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = _punctuation_pattern.sub("", text)
        text = _elongation_pattern.sub(r"\1\1", text)
        processed.append(text)
    return processed

def _replace_preprocessor_with_local(model):
    """Reemplaza el step 'preprocessor' por uno que use full_preprocessing local"""
    try:
        steps = list(model.steps)
    except Exception:
        return model

    new_steps = []
    replaced = False
    for name, estimator in steps:
        if name == "preprocessor" and isinstance(estimator, FunctionTransformer):
            new_steps.append((name, FunctionTransformer(func=full_preprocessing, validate=False)))
            replaced = True
        else:
            new_steps.append((name, estimator))

    if replaced:
        return Pipeline(new_steps)
    else:
        return model

# ===============================
# Carga de modelo y card
# ===============================
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    # reemplazamos el preprocessor por el local estable
    model = _replace_preprocessor_with_local(model)
    return model

def load_model_card():
    with open(MODEL_CARD_PATH, "r", encoding="utf-8") as f:
        card = json.load(f)
    return card

def predict(model, texts: Union[str, List[str]]) -> List[int]:
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (int, float)):
        texts = [str(texts)]
    else:
        texts = [str(t) for t in texts]

    try:
        preds = model.predict(texts)
    except Exception as e:
        raise RuntimeError(f"Error en predict(): {e}") from e

    try:
        return preds.tolist()
    except Exception:
        return list(preds)