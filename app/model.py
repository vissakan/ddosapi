import torch
from drqn_model import DRQN

OBS_DIM = 5
ACTION_DIM = 5
SEQ_LEN = 5

class MitigationModel:
    def __init__(self, model_path="app/model.pth"):
        self.device = torch.device("cpu")
        self.model = DRQN(OBS_DIM, ACTION_DIM)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, observation):
        obs = torch.tensor(observation).float().to(self.device)
        obs = obs.unsqueeze(0).unsqueeze(0)  # batch + seq

        hidden = self.model.init_hidden()
        with torch.no_grad():
            q_values, _ = self.model(obs, hidden)
            action = torch.argmax(q_values).item()

        return action
