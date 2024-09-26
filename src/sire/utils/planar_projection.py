import torch

from src.sire.utils.affine import get_rotation_matrix


def cartesian_to_planar(points: torch.Tensor, normal: torch.Tensor, center: torch.Tensor):
    rot_matrix = get_rotation_matrix(normal.cpu()).to(center)
    planar_points = (torch.linalg.inv(rot_matrix) @ (points.float() - center.float()).T).T

    return planar_points[..., :-1]
