from pydantic import BaseModel
from typing import List, Union

class PredictRequest(BaseModel):
    texts: Union[str, List[str]]

class PredictResponse(BaseModel):
    predictions: List[int]

class ModelInfoResponse(BaseModel):
    model_name: str
    vectorizer: str
    classifier: str
    preprocessing: str
    f1_score: float
    author: str