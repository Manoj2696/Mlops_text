import os
import sys

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
import sys
sys.path.append("/Users/manojyadav/Downloads/AI&DS/MLops/text_classification")


from fastapi import APIRouter, HTTPException
from model.predict import ModelPredictor


router = APIRouter()
predictor = ModelPredictor("model/svm_model.pkl")

@router.post("/predict/")
def predict(text: str):
    try:
        result = predictor.predict(text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
