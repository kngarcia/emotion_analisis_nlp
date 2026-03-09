from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import requests
from API.schemas import PredictRequest, PredictResponse, ModelInfoResponse
from API.inference import load_model, load_model_card, predict

BASE_DIR = Path(__file__).resolve().parent

# ===============================
# FUNCIONES PARA DESCARGA DE ARTEFACTOS
# ===============================
def download_file(url, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"{save_path.name} ya existe, se usa localmente.")
        return
    r = requests.get(url)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded {save_path.name} from GitHub")

def fetch_artifacts():
    """Descarga artefactos desde el repo si no existen"""
    files = {
        "model": "models/best_model.pkl",
        "model_card": "models/model_card_LogReg_C1.0_lbfgs.json",
    }

    repo_base = "https://raw.githubusercontent.com/kngarcia/emotion_analisis_nlp/main/"

    local_paths = {}
    for key, path in files.items():
        url = repo_base + path
        local_path = BASE_DIR / path
        download_file(url, local_path)
        local_paths[key] = local_path
    return local_paths

# ===============================
# INICIALIZAR APP Y CARGAR MODELO
# ===============================
app = FastAPI(title="Sentiment140 Model API")

local_files = fetch_artifacts()
model = load_model()
model_card = load_model_card()

# ===============================
# ENDPOINTS
# ===============================

@app.get("/model_info", response_model=ModelInfoResponse)
def get_model_info():
    return model_card

@app.post("/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest):
    preds = predict(model, req.texts)
    return {"predictions": preds}

@app.get("/ablation_summary")
def ablation_summary():
    csv_path = local_files.get("ablation_csv")
    plot_path = local_files.get("ablation_plot")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    summary_paragraph = (
        "Experimento de ablación: se compararon diferentes configuraciones "
        "y se seleccionó el mejor modelo."
    )
    return {
        "table": df.to_dict(orient="records"),
        "plot": str(plot_path),
        "summary": summary_paragraph
    }

@app.get("/comparison")
def comparison():
    comp_path = local_files.get("comparison")
    try:
        df = pd.read_csv(comp_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    return df.to_dict(orient="records")

@app.get("/work_distribution")
def work_distribution():
    wd_path = local_files.get("work_distribution")
    try:
        df = pd.read_csv(wd_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    return df.to_dict(orient="records")