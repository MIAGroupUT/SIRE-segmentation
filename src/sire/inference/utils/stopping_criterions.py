from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.sire.inference.utils.tracked_vessel import VesselContour
from src.sire.utils.affine import transform_points


class StoppingCriterionBase:
    def __call__(self, **kwargs) -> bool:
        raise NotImplementedError()

    def __str__(self):
        return f"{self.__class__.__name__}()"


class MaxIterationsStoppingCriterion(StoppingCriterionBase):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, iteration: int, **kwargs):
        return self.max_iterations <= iteration

    def __str__(self):
        return f"{self.__class__.__name__}(max_iterations={self.max_iterations})"


class EndPointsStoppingCriterion(StoppingCriterionBase):
    def __init__(self, points: np.array, distance: float = 1):
        self.points = torch.from_numpy(points) if points is not None else None
        self.distance = distance

    def update_points(self, new_points: torch.tensor):
        if self.points is None:
            self.points = new_points
        else:
            self.points = torch.cat([self.points, new_points])

    def __call__(self, point: torch.tensor, **kwargs):
        if self.points is None:
            return False

        distances = torch.linalg.norm(self.points - point.reshape(3), dim=1)
        return torch.any(distances < self.distance).item()

    def __str__(self):
        return f"{self.__class__.__name__}(distance={self.distance})"


class AlreadyTrackedStoppingCriterion(EndPointsStoppingCriterion):
    pass


class PlaneBoundStoppingCriterion(StoppingCriterionBase):
    def __init__(self, center: np.array, normal: np.array, offset: float):
        self.center = torch.from_numpy(center)
        self.normal = torch.from_numpy(normal)
        self.offset = offset

    def __call__(self, point: torch.tensor, **kwargs):
        normal = self.normal / torch.linalg.norm(self.normal)

        point_vec = point - (self.center - self.offset * normal)
        point_vec /= torch.linalg.norm(point_vec)

        cosine_sim = F.cosine_similarity(point_vec.reshape(1, 3), normal.reshape(1, 3)).item()

        return cosine_sim < 0

    def __str__(self):
        return f"{self.__class__.__name__}(normal={self.normal})"


class VesselDiameterStoppingCriterion(StoppingCriterionBase):
    def __init__(self, contour_name: str = "lumen", max_diameter: float = None, min_diameter: float = None):
        self.contour_name = contour_name
        self.max_diameter = max_diameter if max_diameter is not None else np.inf
        self.min_diameter = min_diameter if min_diameter is not None else -np.inf

    def __call__(self, vessel_contour: VesselContour, **kwargs):
        if self.contour_name not in vessel_contour.points.keys():
            return False

        contour = vessel_contour.points[self.contour_name]
        mean_diameter = 2 * torch.linalg.norm(contour - vessel_contour.center, axis=-1).mean().item()

        return (mean_diameter > self.max_diameter) or (mean_diameter < self.min_diameter)

    def __str__(self):
        return f"{self.__class__.__name__}(max_diameter={self.max_diameter}, min_diameter={self.min_diameter})"


class RoiStoppingCriterion(StoppingCriterionBase):
    def __init__(self, roi: np.array):
        self.roi = torch.from_numpy(roi).bool()

    def _in_axes(self, coords: torch.tensor):
        axes = len(self.roi.shape)
        return torch.all(torch.stack([0 <= coords[axis] <= self.roi.shape[axis] for axis in range(axes)]))

    def __call__(self, data: Dict[str, Any], point: torch.tensor, **kwargs):
        affine = data["image_meta_dict"]["affine"]

        point_physical = transform_points(point[None, ...], torch.linalg.inv(affine)).reshape(3)
        point_physical = point_physical.round().long()[[2, 1, 0]]

        if not self._in_axes(point_physical):
            return True
        else:
            try:
                return not self.roi[*point_physical].item()
            except IndexError:
                return True
