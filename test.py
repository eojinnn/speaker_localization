import torch
from model import PGCCPHAT
max_tau = 42

ngcc = PGCCPHAT(max_tau)

# Load the model weights
ngcc.load_state_dict(torch.load(
        "experiments/tdoa_exp/masking_model.pth", map_location=torch.device('cpu')))
ngcc.eval()