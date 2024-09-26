import torch


class EuclideanLoss:
    def __init__(self, p: int = 2):
        self.p = p

    def __call__(self, x: torch.tensor, y: torch.tensor):
        return torch.norm(x - y, p=self.p, dim=-1, keepdim=False).mean()
