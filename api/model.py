import torch


# TODO: change to the new resnet model
def ChessNet() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid(),
    )
