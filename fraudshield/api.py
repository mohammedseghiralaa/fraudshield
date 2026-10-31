from fastapi import FastAPI
from pydantic import create_model

from fraudshield.prediction import predict_transaction

# Input schema: Time, Amount and V1..V28 (all numbers, all required)
fields = {"Time": (float, ...), "Amount": (float, ...)}
fields.update({f"V{i}": (float, ...) for i in range(1, 29)})
Transaction = create_model("Transaction", **fields)

app = FastAPI(title="FraudShield API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    return predict_transaction(transaction.model_dump())
