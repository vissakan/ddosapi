import torch
from app.model import MitigationModel

def test_model_prediction():
    model = MitigationModel("app/model.pth")

    sample_input = [10, 0.1, 100, 5, 20]  # valid observation
    action = model.predict(sample_input)

    assert isinstance(action, int)
    assert 0 <= action <= 4
