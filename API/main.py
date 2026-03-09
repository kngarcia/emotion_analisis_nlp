# main.py
from fastapi import FastAPI, Query
import pandas as pd
from pathlib import Path
from API.schemas import PredictRequest, PredictResponse, ModelInfoResponse
from API.inference import load_model, load_model_card
from API.repo_utils import fetch_artifacts
from fastapi.responses import HTMLResponse
import matplotlib.pyplot as plt
import io
import base64
import json

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Sentiment140 Model API")

# Descargar artefactos (modelo, model card y reportes)
local_files = fetch_artifacts()
model = load_model()
model_card = load_model_card()

# -------------------------
# ENDPOINTS EXISTENTES
# -------------------------
@app.get("/model_info", response_model=ModelInfoResponse)
def get_model_info():
    return model_card

@app.post("/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest):
    preds = model.predict(req.texts)  # asumiendo tu función predict aquí
    return {"predictions": preds}

# -------------------------
# NUEVOS ENDPOINTS DE REPORTES
# -------------------------
@app.get("/ablation_summary_html", response_class=HTMLResponse)
def ablation_summary_html(
    classifier: str | None = Query(None, description="Filtrar por clasificador"),
    vectorizer: str | None = Query(None, description="Filtrar por vectorizador")
):
    # -----------------------------
    # Leer JSON de ablación con runtime incluido
    # -----------------------------
    try:
        with open(local_files["ablation_results"], "r") as f:
            ablation_data = json.load(f)
    except FileNotFoundError:
        return "<h3>No se encontró ablation_results.json</h3>"
    except json.JSONDecodeError as e:
        return f"<h3>Error al leer JSON: {e}</h3>"

    experiments = ablation_data.get("experiments", [])
    summary_text = ablation_data.get("conclusions", "")

    # DataFrame ablation
    df = pd.DataFrame(experiments)

    # -----------------------------
    # Aplicar filtros
    # -----------------------------
    if classifier:
        df = df[df["classifier"] == classifier]
    if vectorizer:
        df = df[df["vectorizer"] == vectorizer]

    # Ordenar por F1-score descendente
    df = df.sort_values(by="f1_score", ascending=False)

    # -----------------------------
    # Crear tabla HTML con F1 y runtime
    # -----------------------------
    table_html = df[["model", "classifier", "vectorizer", "f1_score", "runtime_seconds", "author"]].to_html(
        index=False, classes="table table-striped table-hover", border=0, float_format="%.4f"
    )

    # -----------------------------
    # Gráfica F1-score
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.bar(df["model"], df["f1_score"], color="skyblue")
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("F1 Score")
    plt.title("F1 Score por Modelo")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    f1_img_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" alt="F1 Score Plot"/>'

    # -----------------------------
    # Gráfica Runtime ordenada de mayor a menor
    # -----------------------------
    df_runtime_plot = df.sort_values(by="runtime_seconds", ascending=False)  # ordenar
    plt.figure(figsize=(12,6))
    plt.bar(df_runtime_plot["model"], df_runtime_plot["runtime_seconds"], color="salmon")
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Runtime (s)")
    plt.title("Tiempo de Ejecución por Modelo (Mayor a Menor)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    runtime_img_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" alt="Runtime Plot"/>'
    

    # -----------------------------
    # HTML final
    # -----------------------------
    html_content = f"""
    <html>
    <head>
        <title>Ablation Summary</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
        <style>
            body {{ margin: 20px; font-family: Arial, sans-serif; }}
            h2 {{ color: #2c3e50; }}
            h3 {{ margin-top: 30px; color: #34495e; }}
            p {{ font-size: 14px; }}
        </style>
    </head>
    <body>
        <h2>Resumen de Ablación</h2>
        <p><strong>Filtros aplicados:</strong> Clasificador = {classifier or 'Todos'}, Vectorizador = {vectorizer or 'Todos'}</p>
        {table_html}
        <h3>Gráfica F1-score por Modelo</h3>
        {f1_img_html}
        <h3>Gráfica Runtime por Modelo</h3>
        {runtime_img_html}
        <h3>Conclusiones</h3>
        <p>{summary_text}</p>
    </body>
    </html>
    """
    return html_content

@app.get("/comparison_html", response_class=HTMLResponse)
def comparison_html():
    # Leer JSON de ablación
    try:
        with open(local_files["ablation_results"], "r") as f:
            ablation_data = json.load(f)
    except FileNotFoundError:
        return "<h3>No se encontró ablation_results.json</h3>"
    except json.JSONDecodeError as e:
        return f"<h3>Error al leer JSON: {e}</h3>"

    # Extraer el mejor modelo y HuggingFace DistilBERT
    best_model = ablation_data.get("best_model", {})
    experiments = ablation_data.get("experiments", [])
    hf_model = next((e for e in experiments if "DistilBERT" in e["model"]), None)

    if not hf_model:
        return "<h3>No se encontró el modelo HuggingFace DistilBERT en los experimentos.</h3>"

    # Crear DataFrame comparativo
    df = pd.DataFrame([best_model, hf_model])
    df["F1 Score (%)"] = df["f1_score"] * 100
    table_html = df[["model", "classifier", "vectorizer", "F1 Score (%)", "author"]].to_html(
        index=False, classes="table table-striped table-hover", border=0
    )

    # Crear gráfica de barras comparativa
    plt.figure(figsize=(6,4))
    plt.bar(df["model"], df["f1_score"], color=["skyblue", "orange"])
    plt.ylabel("F1 Score")
    plt.ylim(0,1)
    plt.title("Comparación Mejor Modelo vs HuggingFace DistilBERT")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{img_base64}" alt="Comparison Plot"/>'

    # Conclusiones automáticas
    conclusions_html = f"""
    <ul>
        <li>El modelo <strong>{best_model['model']}</strong> alcanza un F1-score de {best_model['f1_score']:.3f}, siendo el mejor en este dataset.</li>
        <li>El modelo <strong>{hf_model['model']}</strong> alcanza un F1-score de {hf_model['f1_score']:.3f}, significativamente inferior.</li>
        <li>DistilBERT no fue entrenado con datos de Twitter, lo que explica su rendimiento más bajo en este dataset.</li>
        <li>Los modelos lineales y TF-IDF optimizados siguen siendo los más efectivos para este tipo de clasificación de sentimiento.</li>
    </ul>
    """

    # HTML final
    html_content = f"""
    <html>
    <head>
        <title>Comparación Modelos</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
        <style>
            body {{ margin: 20px; font-family: Arial, sans-serif; }}
            h2 {{ color: #2c3e50; }}
            h3 {{ margin-top: 30px; color: #34495e; }}
            p, li {{ font-size: 14px; }}
        </style>
    </head>
    <body>
        <h2>Comparación Mejor Modelo vs HuggingFace DistilBERT</h2>
        {table_html}
        <h3>Gráfica Comparativa</h3>
        {img_html}
        <h3>Conclusiones</h3>
        {conclusions_html}
    </body>
    </html>
    """
    return html_content

@app.get("/work_distribution_html", response_class=HTMLResponse)
def work_distribution_html():
    # -----------------------------
    # Leer JSON
    # -----------------------------
    try:
        with open(local_files["ablation_results"], "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return "<h3>No se encontró ablation_results.json</h3>"
    except json.JSONDecodeError as e:
        return f"<h3>Error al leer JSON: {e}</h3>"

    work_distribution = data.get("work_distribution", [])

    if not work_distribution:
        return "<h3>No hay información de distribución de trabajo</h3>"

    # -----------------------------
    # Crear DataFrame
    # -----------------------------
    df = pd.DataFrame(work_distribution)
    df = df.sort_values(by="experiments", ascending=False)  # ordenar de mayor a menor

    # -----------------------------
    # Crear tabla HTML
    # -----------------------------
    table_html = df.to_html(index=False, classes="table table-striped table-hover", border=0)

    # -----------------------------
    # Crear gráfica de barras
    # -----------------------------
    plt.figure(figsize=(8,4))
    plt.bar(df["author"], df["experiments"], color="mediumseagreen")
    plt.ylabel("Número de Experimentos")
    plt.title("Distribución de Trabajo por Autor")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{img_base64}" alt="Work Distribution Plot"/>'

    # -----------------------------
    # HTML final
    # -----------------------------
    html_content = f"""
    <html>
    <head>
        <title>Distribución de Trabajo</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
        <style>
            body {{ margin: 20px; font-family: Arial, sans-serif; }}
            h2 {{ color: #2c3e50; }}
            h3 {{ margin-top: 30px; color: #34495e; }}
            p {{ font-size: 14px; }}
        </style>
    </head>
    <body>
        <h2>Distribución de Trabajo</h2>
        {table_html}
        <h3>Gráfica de Experimentos por Autor</h3>
        {img_html}
    </body>
    </html>
    """
    return html_content
