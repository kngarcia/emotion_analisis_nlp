# Sentiment140 — Sentiment Analysis Laboratory

## Overview

This project implements a **sentiment analysis system** for Twitter data using the **Sentiment140** dataset. The goal is to explore how far classical machine learning methods can approach or outperform the performance of a Transformer-based model (**DistilBERT** fine-tuned on SST-2).

The study is conducted through a systematic **ablation analysis**, evaluating the impact of:

* Preprocessing techniques
* Text encoding strategies
* Different machine learning models

All experiments are tracked using **MLflow**, and the final model is deployed through a **FastAPI** inference service.

---

## Dataset

* **Source:** HuggingFace Datasets (`adilbekovich/Sentiment140Twitter`)
* **Characteristics:**

| Property | Value |
| --- | --- |
| **Domain** | Twitter |
| **Task** | Binary Sentiment Classification |
| **Classes** | Positive / Negative |
| **Training samples** | 1,600,000 |
| **Test samples** | 498 |

**Example tweet:** *"I love this movie so much!!"*
**Labels:** `1` → Positive | `0` → Negative

---

## Project Goal

The main objective is to obtain a sentiment classification model using classical machine learning techniques with performance comparable to the HuggingFace Transformers pipeline (DistilBERT).

### Primary Metric

> **F1 Score (weighted)**

---

## Baseline Model

The baseline configuration follows the instructions from the laboratory specification.

**Pipeline:**
Text → Minimal cleaning → Bag of Words (CountVectorizer) → Multinomial Naive Bayes → Prediction

**Cleaning steps:**

1. Lowercasing
2. Remove URLs
3. Remove hashtags
4. Remove user mentions

**Baseline components:**

* **Vectorizer:** Bag of Words
* **Model:** Multinomial Naive Bayes
* **Dataset:** Sentiment140
* **Metric:** F1 Score

---

## Expected Performance (Reference Model)

To establish a reference performance, the project evaluates the HuggingFace pipeline: `pipeline("sentiment-analysis")`.

* **Default model:** `distilbert-base-uncased-finetuned-sst-2-english`
* **Considerations:** This model was trained on the SST-2 dataset, not specifically on Twitter data.

---

## Experiment Tracking

All experiments are tracked using an **MLflow Tracking Server**. Each run records:

* **Parameters:** Preprocessing config, encoding method, model type, and hyperparameters.
* **Metrics:** F1 score and runtime.
* **Tags:**
* `author=Nicolas` (or specific member name)
* `dataset`, `model_type`, `feature_type`, `experiment_stage`.



---

## Ablation Study

The project performs an ablation analysis organized into three groups to quantify the contribution of each component.

### A. Preprocessing Experiments (spaCy)

**Pipeline:** `tokenization` → `normalization` → `lemmatization` → `filtering`

| Component | Options |
| --- | --- |
| **Lemmatization** | on / off |
| **Stopwords** | keep / drop |
| **Emojis** | keep / translate / drop |
| **Punctuation** | keep / drop |
| **Elongation normalization** | on / off (e.g., *goooood* → *good*) |

### B. Encoding Experiments

Using the best preprocessing configuration from Stage A:

1. **TF-IDF (Unigrams):** `TfidfVectorizer(ngram_range=(1,1))`
2. **TF-IDF (Bigrams):** `TfidfVectorizer(ngram_range=(1,2))`

### C. Model Experiments

Using the best preprocessing and encoding configurations:

| Model Type | Example |
| --- | --- |
| **Tree-based** | Random Forest |
| **Parametric** | Logistic Regression |
| **Neural Network** | Dense Neural Network |

**Neural Network Architecture:**

`Input` → `Dense(512)` → `Dropout` → `Dense(128)` → `Dropout` → `Dense(1, sigmoid)`

---

## System Architecture (AWS)

The system is deployed using three main components:

1. **EC2-A — MLflow Tracking Server:** Stores experiment runs, metrics, and artifacts. (No training occurs here).
2. **SageMaker Notebook (m5.xlarge):** Used by each member to run experiments and connect to the remote MLflow server.
3. **EC2-B — FastAPI Inference API:** Exposes the best model (preprocessing + encoding + model) for predictions.

---

## API Endpoints

* `GET /model_info`: Returns model description.
* `POST /predict`: Performs single or batch sentiment prediction.
* `GET /ablation_summary`: Returns results table and performance plots.
* `GET /comparison`: Compares the best classical model vs. DistilBERT (F1 and Latency).
* `GET /work_distribution`: Displays experiments per team member.

---

## Project Structure

```text
sentiment140/
│
├─ data/
│  ├─ raw/
│  └─ processed/
│
├─ notebooks/
│  ├─ 01_preprocessing.ipynb
│  ├─ 04_models_nicolas.ipynb
│  └─ ...
│
├─ models/
│  ├─ best_model.pkl
│  └─ best_model_card.json
│
└─ api/
   └─ main.py

```

---

## Authors

| Member | Experiments |
| --- | --- |
| **Nicolas** | Baseline + Preprocessing + Neural Network |
| **Kevin** | Encoding + Tree models + Logistic Regression |

