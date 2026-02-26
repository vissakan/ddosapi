from fastapi import FastAPI
from pydantic import BaseModel
from model import MitigationModel

app = FastAPI()

model = MitigationModel()

class Metrics(BaseModel):
    observation: list

@app.post("/predict")
def predict(metrics: Metrics):
    action = model.predict(metrics.observation)
    return {"mitigation_action": action}

