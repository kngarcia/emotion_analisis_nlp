Sentiment140 — Sentiment Analysis Laboratory
Overview

This project implements a sentiment analysis system for Twitter data using the Sentiment140 dataset.
The goal is to explore how far classical machine learning methods can approach or outperform the performance of a Transformer-based model (DistilBERT fine-tuned on SST-2).

The study is conducted through a systematic ablation analysis, evaluating the impact of:

preprocessing techniques

text encoding strategies

different machine learning models

All experiments are tracked using MLflow, and the final model is deployed through a FastAPI inference service.

Dataset

Dataset source:

HuggingFace Datasets

adilbekovich/Sentiment140Twitter

Dataset characteristics:

Property	Value
Domain	Twitter
Task	Binary Sentiment Classification
Classes	Positive / Negative
Training samples	1,600,000
Test samples	498

Example tweet:

I love this movie so much!!

Label:

1 → Positive
0 → Negative
Project Goal

The main objective of this laboratory is:

To obtain a sentiment classification model using classical machine learning techniques with performance comparable to the HuggingFace Transformers pipeline (DistilBERT).

The evaluation is based on a single agreed metric:

Primary metric

F1 Score (weighted)
Baseline Model

The baseline configuration follows the instructions from the laboratory specification.

Pipeline:

Text
↓
Minimal cleaning
↓
Bag of Words (CountVectorizer)
↓
Multinomial Naive Bayes
↓
Prediction

Cleaning steps:

lowercasing

remove URLs

remove hashtags

remove user mentions

Baseline components:

Component	Configuration
Vectorizer	Bag of Words
Model	Multinomial Naive Bayes
Dataset	Sentiment140
Metric	F1 Score
Expected Performance (Reference Model)

To establish a reference performance, the project evaluates the HuggingFace pipeline:

pipeline("sentiment-analysis")

Default model:

distilbert-base-uncased-finetuned-sst-2-english

Important considerations:

trained on SST-2 dataset

not trained specifically on Twitter data

This comparison allows evaluating whether classical methods can compete with transformer models on this dataset.

Experiment Tracking

All experiments are tracked using:

MLflow Tracking Server

Each run records:

Parameters

preprocessing configuration

encoding method

model type

hyperparameters

Metrics

F1 score

runtime

Tags

Each experiment includes a tag identifying the author:

author=<member_name>

Other tags include:

dataset
model_type
feature_type
experiment_stage
Ablation Study

The project performs an ablation analysis to quantify the contribution of each component.

The experiments are organized into three groups.

A. Preprocessing Experiments (spaCy)

Pipeline order:

tokenization → normalization → lemmatization → filtering

Tested configurations:

Component	Options
Lemmatization	on / off
Stopwords	keep / drop
Emojis	keep / translate / drop
Punctuation	keep / drop
Elongation normalization	on / off

Example:

goooood → good

These experiments use:

Baseline model (MultinomialNB)
Baseline encoding (Bag of Words)

The goal is to measure the individual impact of preprocessing components.

B. Encoding Experiments

Using the best preprocessing configuration obtained in the previous stage.

Encoding strategies tested:

TF-IDF (Unigrams)
TfidfVectorizer(ngram_range=(1,1))
TF-IDF (Bigrams)
TfidfVectorizer(ngram_range=(1,2))

Goal:

Evaluate whether TF-IDF improves performance compared to Bag of Words.

C. Model Experiments

Using the best preprocessing and encoding configuration.

Models tested belong to different families:

Model Type	Example
Tree-based	Random Forest
Parametric	Logistic Regression
Neural Network	Dense Neural Network

Example neural network architecture:

Input
↓
Dense(512)
↓
Dropout
↓
Dense(128)
↓
Dropout
↓
Dense(1, sigmoid)
D. Comparison with DistilBERT

The best classical model is compared against:

HuggingFace Transformers pipeline

Evaluation criteria:

Metric	Description
F1 Score	classification performance
Runtime	inference latency

Important note:

DistilBERT was not trained on Twitter data, which may influence its performance.

Project Structure
sentiment140/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .env.example
│
├─ data/
│ ├─ raw/
│ ├─ processed/
│ ├─ encoded/
│
├─ notebooks/
│ ├─ 01_preprocessing.ipynb
│ ├─ 02_encoding.ipynb
│ ├─ 03_baseline.ipynb
│ ├─ 04_models_<member>.ipynb
│ └─ 05_distilbert.ipynb
│
├─ models/
│ ├─ best_model.pkl
│ └─ best_model_card.json
│
└─ api/
   ├─ main.py
   ├─ schemas.py
   ├─ inference.py
   └─ requirements.txt
System Architecture (AWS)

The system is deployed using three main components.

EC2-A — MLflow Tracking Server

Purpose:

store experiment runs

store metrics and artifacts

Important rule:

No model training occurs on this machine.

SageMaker Notebook (per group member)

Purpose:

run experiments

connect to remote MLflow tracking server

Instance type:

m5.xlarge
EC2-B — FastAPI Inference API

Purpose:

Expose the best model obtained from the ablation study.

The API loads a pipeline containing:

preprocessing
encoding
model
API Endpoints
GET /model_info

Returns model description.

Example response:

{
 "model": "Neural Network",
 "encoding": "TF-IDF",
 "preprocessing": "lemmatization + stopword removal",
 "metric": "F1 Score"
}
POST /predict

Performs sentiment prediction.

Supports:

single text

batch predictions

Example request:

{
 "texts": [
   "I love this movie!",
   "This product is terrible"
 ]
}
GET /ablation_summary

Returns the ablation analysis including:

results table

performance plot

short conclusions

GET /comparison

Compares:

Best classical model
vs
HuggingFace DistilBERT

Metrics reported:

F1 Score

inference runtime

GET /work_distribution

Displays the distribution of experiments across team members.

Example:

Member	Experiments
Nicolas	Baseline + Neural Network
Member 2	Preprocessing
Member 3	Tree models
Installation

Clone repository:

git clone https://github.com/<repo>/sentiment140.git

Install dependencies:

pip install -r requirements.txt
Running Experiments

Experiments are executed in SageMaker notebooks.

Example workflow:

01_preprocessing.ipynb
↓
02_encoding.ipynb
↓
03_baseline.ipynb
↓
04_models_<member>.ipynb
↓
05_distilbert.ipynb
Final Model

The best model obtained from the ablation study is saved in:

models/best_model.pkl

Model documentation:

models/best_model_card.json
Authors

Group members:

Member	Experiments
Nicolas	Baseline + Neural Network
Member 2	Preprocessing
Member 3	Tree models
License

Academic project for educational purposes.
