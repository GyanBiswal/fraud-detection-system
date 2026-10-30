from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predict import load_model, predict_transaction

app = FastAPI(title="Transaction Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = load_model()


class TransactionRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: TransactionRequest):
    try:
        if request.features.get("Amount", 0) < 0:
            raise ValueError("Amount cannot be negative")
        return predict_transaction(request.features, bundle)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))