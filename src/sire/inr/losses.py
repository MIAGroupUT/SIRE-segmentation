from typing import Dict

import torch
import torch.nn.functional as F

from torch_cluster import knn


def gradient(y: torch.tensor, x: torch.tensor, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)

    grad = torch.autograd.grad(
        y, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    return grad


def divergence(points: torch.tensor, gradients: torch.tensor):
    nonmnfld_dx = gradient(gradients[:, 0], points)
    nonmnfld_dy = gradient(gradients[:, 1], points)
    nonmnfld_dz = gradient(gradients[:, 2], points)

    nonmnfld_divergence = nonmnfld_dx[:, 0] + nonmnfld_dy[:, 1] + nonmnfld_dz[:, 2]
    nonmnfld_divergence[nonmnfld_divergence.isnan()] = 0

    return torch.clamp(torch.abs(nonmnfld_divergence), 0.1, 50)


class ManifoldLoss:
    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        return (data["on"]["sdf"] ** 2).mean()


class NonManifoldLoss:
    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        return torch.exp(-100 * torch.abs(data["off"]["sdf"])).mean()


class NeumannLoss:
    """
    Normal alignment loss
    https://arxiv.org/abs/2309.01793#:~:text=In%20accordance%20with%20Differential%20Geometry,for%20points%20near%20the%20surface.
    """

    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        grad_on = gradient(data["on"]["sdf"], data["on"]["coords"])
        grad_on = grad_on / torch.linalg.norm(grad_on, dim=1, keepdims=True)

        return (1 - torch.einsum("ij,ij->i", grad_on, data["on"]["normals"])).mean()


class SkeletonLoss:
    """
    Motivated by: https://doi.org/10.1016/j.cag.2023.06.012
    Skeleton of SDF is formed by singularities for which gradient should be 0.
    """

    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        skeleton_grad = gradient(data["skeleton"]["sdf"], data["skeleton"]["coords"])
        return (skeleton_grad**2).mean()


class EikonalLoss:
    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        grad_off = gradient(data["off"]["sdf"], data["off"]["coords"])
        grad_on = gradient(data["on"]["sdf"], data["on"]["coords"])
        grad = torch.cat([grad_on, grad_off], dim=0)

        return torch.pow(1 - torch.linalg.norm(grad, dim=1), 2).mean()


class TVLoss:
    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        surface_grad_norm = gradient(data["off"]["sdf"], data["off"]["coords"]).norm(dim=1)
        surface_grad_norm_grad = gradient(surface_grad_norm, data["off"]["coords"])

        return surface_grad_norm_grad.norm(dim=1).mean()


class NeuralPullLoss:
    def __call__(self, points: torch.tensor, data: Dict[str, torch.tensor]):
        _, closest_idx = knn(points, data["off"]["coords"], k=1)
        nearest_coords = points[closest_idx.detach()]

        grad_off = gradient(data["off"]["sdf"], data["off"]["coords"])
        grad_norm = F.normalize(grad_off, dim=1)

        moved_coords = data["off"]["coords"].detach() - grad_norm * data["off"]["sdf"]

        return torch.linalg.norm((nearest_coords - moved_coords), ord=2, dim=-1).mean()
