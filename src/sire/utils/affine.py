from typing import Dict, Union

import numpy as np
import torch

from pytransform3d.rotations import matrix_from_axis_angle
from scipy.spatial.transform import Rotation as Rot


def get_affine(data):
    """
    Get or construct the affine matrix of the image, it can be used to correct
    spacing, orientation or execute spatial transforms.
    Args:
    data: an ITK image object loaded from an image file or props dictionary.

    """
    if isinstance(data, Dict):
        direction = np.array(data["sitk_stuff"]["direction"]).reshape(3, 3)
        spacing = np.array(data["sitk_stuff"]["spacing"])
        origin = np.array(data["sitk_stuff"]["origin"])

    else:
        direction = np.array(data.GetDirection()).reshape(3, 3)
        spacing = np.asarray(data.GetSpacing())
        origin = np.asarray(data.GetOrigin())

    sr = min(max(direction.shape[0], 1), 3)
    affine: np.ndarray = np.eye(sr + 1)
    affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
    affine[:sr, -1] = origin[:sr]
    return torch.from_numpy(affine).float(), torch.from_numpy(spacing).float()


def cart2spher(cart_coords: Union[torch.Tensor, np.array]):
    # transform an Nx3 matrix of (unit length) Cartesian coordinates
    # into an Nx3 matrix with [r, phi, theta] spherical coordinates.
    # normalize cartesian coordinates
    if torch.is_tensor(cart_coords):
        cart_coords /= torch.linalg.norm(cart_coords, dim=1, keepdim=True)
        coords_spherical = torch.ones_like(cart_coords)
        theta = torch.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = torch.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).long()
    else:
        cart_coords = cart_coords / np.expand_dims(np.linalg.norm(cart_coords, axis=1), 1)
        coords_spherical = np.ones_like(cart_coords)
        theta = np.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = np.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).astype(int)

    coords_spherical[:, 1] = phi
    coords_spherical[:, 2] = theta
    return coords_spherical


def transform_points(points: torch.tensor, affine: torch.tensor):
    points = torch.cat([points, torch.ones([points.shape[0], 1]).to(points)], dim=1).T
    return (affine.float() @ points.float()).T[:, :-1]


def get_rotation_matrix(vector):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    Adapted from: https://stackoverflow.com/questions/63287960/python-rotate-plane-set-of-points-to-match-new-normal-vector-using-scipy-spat
    """
    vector = vector / torch.linalg.norm(vector)
    axis = np.cross(np.array([0, 0, 1]), vector.numpy())

    # determine angle between new normal and z-axis
    dot_product = np.dot(np.array([0, 0, 1]), vector.numpy())
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    a = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(a)
    M = Rot.from_matrix(R).as_matrix()
    return torch.from_numpy(M).float()
