# Sentiment140 Inference API

## Overview

This API exposes the best sentiment classification model obtained from the Sentiment140 ablation study through a **FastAPI** service. The system allows for sentiment prediction on Twitter text and provides access to experimental results and project metadata.

The API loads a trained machine learning pipeline that includes:

* **Text preprocessing**
* **Feature encoding**
* **Classification model**

---

## System Architecture Context

This API corresponds to the **EC2-B** inference service within the laboratory ecosystem.

### Workflow

1. **SageMaker Notebooks:** Train models and perform ablation.
2. **MLflow Tracking Server (EC2-A):** Logs all experiments and metrics.
3. **Git Repository:** Stores the finalized `best_model.pkl` and `model_card.json`.
4. **EC2-B FastAPI:** Pulls artifacts from Git (if not local) and loads the model pipeline.
5. **REST API Endpoints:** Serves predictions to the end user.

---

## Model Pipeline

The deployed model is a Scikit-learn Pipeline structured as follows:

**Raw Text** → **Text Preprocessing** → **Vectorization** → **Classifier**

### Preprocessing Steps

The pipeline applies specific cleaning logic to ensure data consistency:

* Lowercasing
* URL, User mention, and Hashtag removal
* Punctuation removal
* **Elongation normalization** (e.g., *goooood* → *good*)

---

## Model Artifacts

The API automatically manages the following files:

* `models/best_model.pkl`: The serialized inference pipeline.
* `models/model_card_LogReg_C1.0_lbfgs.json`: Metadata describing the selected model.

---

## API Endpoints

### 1. `GET /model_info`

Returns metadata describing the selected model, retrieved from the model card.

**Response Example:**

```json
{
  "model_name": "Logistic Regression",
  "vectorizer": "TF-IDF",
  "classifier": "LogisticRegression",
  "preprocessing": "lowercase + url removal + mention removal + punctuation removal",
  "f1_score": 0.86,
  "author": "Nicolas"
}

```

| Field | Description |
| --- | --- |
| **model_name** | Name of the selected model |
| **vectorizer** | Text encoding method |
| **f1_score** | Performance metric |
| **author** | Team member responsible for the model |

---

### 2. `POST /predict`

Performs sentiment prediction. Supports single strings or batches of text.

**Request Body:**

```json
{
  "texts": ["I love this movie!", "This product is terrible"]
}

```

**Response:**

```json
{
  "predictions": [1, 0]
}

```

* **0:** Negative
* **1:** Positive

---

### 3. `GET /ablation_summary`

Returns a summary of the ablation study, including results and visualization paths.

---

### 4. `GET /comparison`

Compares the best classical ML model against the HuggingFace Transformer baseline (DistilBERT) based on **F1 Score** and **Inference Runtime**.

---

### 5. `GET /work_distribution`

Provides transparency regarding team contributions.

**Example Response:**
| Member | Experiments |
| :--- | :--- |
| **Nicolas** | Baseline + Neural Network |
| **Member2** | Preprocessing experiments |
| **Member3** | Tree-based models |

---

## Implementation Details

### Input Validation

Requests are validated using **Pydantic** schemas:

```python
class PredictRequest(BaseModel):
    texts: Union[str, List[str]]

class PredictResponse(BaseModel):
    predictions: List[int]

```

### Running the API

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run the server:** `uvicorn API.main:app --host 0.0.0.0 --port 8000`
3. **Interactive Documentation:** Access Swagger UI at `http://ec2-52-90-75-165.compute-1.amazonaws.com:8000/docs` 

---

### Key Design Decisions

* **Model Loading:** Handled via `pickle`. Preprocessing steps are integrated into the pipeline to ensure reproducibility.
* **Stateless Deployment:** Artifacts are retrieved from Git if missing, making the EC2 instance easy to replace or scale.
