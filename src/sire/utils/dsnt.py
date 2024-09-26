r"""Implementation of "differentiable spatial to numerical" (soft-argmax) operations, as described in the paper
"Numerical Coordinate Regression with Convolutional Neural Networks" by Nibali et al. for 1D and 3D case based on
Kornia implementation of 2D case: https://github.com/kornia/kornia/blob/master/kornia/geometry/subpix/dsnt.py
"""
from typing import List, Sequence, Union

import torch

from kornia.core import softmax
from kornia.geometry.subpix.dsnt import _safe_zero_division
from kornia.utils.grid import create_meshgrid, create_meshgrid3d


def normalize_coords(pos: torch.tensor, shape: Union[int, Sequence[int]]):
    if isinstance(shape, int):
        shape = [shape]

    return torch.stack([(pos[..., i] / (s - 1) - 0.5) * 2 for i, s in enumerate(shape)], dim=-1)


def unnormalize_coords(pos: torch.tensor, shape: Union[int, Sequence[int]]):
    if isinstance(shape, int):
        shape = [shape]

    return torch.stack([((pos[..., i] / 2) + 0.5) * (s - 1) for i, s in enumerate(shape)], dim=-1)


def create_meshgrid1d(size: int, normalized_coordinates: bool = True):
    pos = torch.linspace(0, size - 1, size).view(-1, 1)

    if normalized_coordinates:
        pos = (pos / (size - 1) - 0.5) * 2

    return pos


def render_gaussian(
    mean: torch.Tensor, std: torch.Tensor, size: List[int], normalized_coordinates: bool = True
) -> torch.Tensor:
    """
    mean: (B, C, d)
    std: (d)
    size: (d)
    return: (B, C, D, W, H), (B, C, W, H), (B, C, H)
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    batch_size, channels, dim = mean.shape
    extend_shape = torch.ones(dim, dtype=mean.dtype).int()

    # Create coordinates grid
    if dim == 3:
        meshgrid_func = create_meshgrid3d
    elif dim == 2:
        meshgrid_func = create_meshgrid
    elif dim == 1:
        meshgrid_func = create_meshgrid1d
    else:
        raise NotImplementedError(f"Function not supported for dim '{dim}'. Supported dims: [1, 2, 3].")

    grid: torch.Tensor = meshgrid_func(*size, normalized_coordinates)
    grid = grid.to(device=mean.device, dtype=mean.dtype)

    # Build n-dimensional gaussian
    pos: torch.Tensor = grid.view(*size, -1)
    dist = (pos - mean.reshape(batch_size, channels, *extend_shape, dim)) ** 2
    k = -0.5 * torch.reciprocal(std.reshape(*extend_shape, dim))

    exps = torch.exp(dist * k)
    gauss = exps.prod(-1)

    # Rescale so that values sum to one.
    val_sum = gauss.reshape(batch_size, channels, -1).sum(-1)
    val_sum = val_sum.reshape(batch_size, channels, *extend_shape)
    gauss = _safe_zero_division(gauss, val_sum)

    return gauss


def spatial_softmax(input: torch.Tensor, temperature: torch.Tensor = torch.tensor(1.0)):
    """
    input: (B, C, D, W, H), (B, C, W, H), (B, C, H)
    return: (B, C, D, W, H), (B, C, W, H), (B, C, H)
    """
    batch_size, channels = input.shape[:2]
    dims = input.shape[2:]

    x: torch.Tensor = input.view(batch_size, channels, -1)
    x_soft: torch.Tensor = softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, *dims)


def spatial_expectation(input: torch.Tensor, normalized_coordinates: bool = True):
    """
    input: (B, C, D, W, H), (B, C, W, H), (B, C, H)
    return: (B, C, d)
    """
    batch_size, channels = input.shape[:2]
    dims = input.shape[2:]
    dim = len(dims)

    # Create coordinates grid
    if dim == 3:
        meshgrid_func = create_meshgrid3d
    elif dim == 2:
        meshgrid_func = create_meshgrid
    elif dim == 1:
        meshgrid_func = create_meshgrid1d
    else:
        raise NotImplementedError(f"Function not supported for dim '{dim}'. Supported dims: [1, 2, 3].")

    # Create coordinates grid.
    grid: torch.Tensor = meshgrid_func(*dims, normalized_coordinates)
    grid = grid.to(device=input.device, dtype=input.dtype)

    input_flat: torch.Tensor = input.view(batch_size, channels, -1)
    output: torch.Tensor = torch.cat(
        [torch.sum(grid[..., i].reshape(-1) * input_flat, -1, keepdim=True) for i in range(dim)], -1
    )

    return output.view(batch_size, channels, -1)
