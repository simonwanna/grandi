import torch


# NOTE: make sure that this matches the trained model
def ChessNet() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid(),
    )
