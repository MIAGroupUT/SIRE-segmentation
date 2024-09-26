import copy

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
)

import numpy as np
import torch

from monai.transforms import Randomizable, Transform
from pytorch_lightning.utilities import move_data_to_device
from sklearn.metrics.pairwise import haversine_distances
from torch.nn.functional import grid_sample

from src.sire.utils.affine import transform_points
from src.sire.utils.tracker import TrackerSphere, cart2spher


class SampleSIRE(Transform, Randomizable):
    def __init__(
        self,
        npoints: int = 32,
        alpha: float = 3,
        r: float = 0.3,
        subdivisions: int = 3,
        center_noise: float = 0,
        num_samples: int = None,
        stratify_radius: bool = False,
        device: str = "cpu",
    ):
        self.npoints = npoints
        self.alpha = alpha
        self.r = r
        self.center_noise = torch.tensor(center_noise).repeat(3)
        self.num_samples = num_samples
        self.stratify_radius = stratify_radius
        self.device = device

        self.sphere = TrackerSphere(subdivisions=subdivisions)
        self.sphereverts = torch.from_numpy(self.sphere.sphereverts).float()[:, 1:] - torch.tensor([np.pi / 2, 0])

    def _get_spheres(
        self,
        image: torch.tensor,
        affine: torch.tensor,
        spheres: torch.tensor,
        center: torch.tensor,
    ):
        # Convert center to image coordinates
        image_center = transform_points(center[None, :], torch.linalg.inv(affine))

        # Compute casting rays for spheres
        rays = spheres.coords + image_center
        rays = (rays * (2 / torch.flip(torch.tensor(image.shape).to(self.device), dims=[0])) - 1).float()

        # Sample features from the underlying image
        spheres = copy.deepcopy(spheres)
        spheres.center = image_center
        spheres.features = (
            grid_sample(
                image[None, None, ...],
                rays.view(1, 1, 1, rays.shape[0], 3),
                padding_mode="reflection",
                align_corners=True,
            )
            .squeeze()
            .reshape(-1, self.npoints)
        )

        return spheres

    def __call__(self, data: Dict[str, Any], point: torch.tensor, labels: Tuple[str] = ("label",)):
        data = move_data_to_device(data, self.device)
        point = move_data_to_device(point, self.device)

        id = data["id"]
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        scales = data["scales_meta_dict"]["scales"]
        nverts = data["scales_meta_dict"]["nverts"]
        spheres = copy.deepcopy(data["scales"])

        sampled_spheres = self._get_spheres(image, affine, spheres, point)

        return {
            "global": {
                "id": id,
                "labels": labels,
                "image": image,
                "affine": affine,
                "scales": scales,
                "nverts": nverts,
            },
            "sample": {
                "spheres": sampled_spheres,
                "center": point,
                "index": torch.tensor([0]),
                "contours": torch.zeros(1, len(labels), 3),
            },
        }


class SampleSIRESegmentation(SampleSIRE):
    def __init__(
        self,
        contour_types: List[str],
        npoints: int = 32,
        alpha: float = 3,
        r: float = 0.3,
        subdivisions: int = 3,
        center_noise: float = 1,
        num_samples: int = None,
        stratify_radius: bool = False,
    ):
        super().__init__(npoints, alpha, r, subdivisions, center_noise, num_samples, stratify_radius)
        self.contour_types = contour_types
        self.current_epoch = 0

    def _generate_noise(self, normal: torch.tensor):
        noise = torch.normal(torch.zeros(3), std=self.center_noise)
        noise_proj = noise - normal * np.dot(noise, normal)

        return noise_proj

    def _get_sample(self, data: Dict[str, Any], index: int, contour_type: int):
        data = move_data_to_device(data, self.device)

        # Global pointers
        id = data["id"]
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        scales = data["scales_meta_dict"]["scales"]
        nverts = data["scales_meta_dict"]["nverts"]

        # Local copies
        spheres = copy.deepcopy(data["scales"])
        center = copy.deepcopy(data["contour"][contour_type]["center"][index])
        normal = copy.deepcopy(data["contour"][contour_type]["normal"][index])
        branch = copy.deepcopy(data["contour"][contour_type]["branch"][index])

        if (points := data["contour"][contour_type]["points"][index]) is not None:
            points = copy.deepcopy(points).reshape(-1, 1, 3)
        else:
            points = torch.zeros(1, 1, 3)

        # Apply random noise to center and sample spheres
        center = copy.deepcopy(data["contour"][contour_type]["center"][index])
        noise_center = center + self._generate_noise(normal)
        sampled_spheres = self._get_spheres(image, affine, spheres, noise_center)

        return {
            "global": {
                "id": id,
                "contour_types": self.contour_types,
                "image": image,
                "affine": affine,
                "scales": scales,
                "nverts": nverts,
            },
            "sample": {
                "spheres": sampled_spheres,
                "label": contour_type,
                "index": torch.tensor(index),
                "normal": normal,
                "center": noise_center,
                "contours": points,
                "branch": branch,
            },
        }

    def __call__(self, data: Dict[str, Any]):
        if self.current_epoch < 0:
            num_all_samples = len(data["contour"]["lumen"]["center"])
            replace = self.num_samples > num_all_samples

            indices = self.R.choice(torch.arange(num_all_samples), size=self.num_samples, replace=replace)
            all_data = [self._get_sample(data, indice, "lumen") for indice in indices]

        else:
            all_data = []

            for contour_type in self.contour_types:
                num_all_samples = len(data["contour"][contour_type]["center"])

                if self.num_samples is None:
                    indices = torch.arange(num_all_samples)
                else:
                    contour_type_nsamples = self.num_samples // len(self.contour_types)

                    replace = contour_type_nsamples > num_all_samples
                    indices = self.R.choice(torch.arange(num_all_samples), size=contour_type_nsamples, replace=replace)

                all_data.extend([self._get_sample(data, indice, contour_type) for indice in indices])

        self.R.shuffle(all_data)

        return all_data


class SampleSIRETracker(SampleSIRE):
    def _get_point(self, centerline_function: Callable, index: int):
        stepsize = centerline_function(index)[-1] * 0.25
        center = centerline_function(index)[:3]
        neighbours = centerline_function([index - stepsize, index + stepsize]).T[:, :3]

        tangents = neighbours - center
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        return torch.from_numpy(center), torch.from_numpy(tangents)

    def _get_direction_heatmap(self, directions: torch.tensor):
        directions = cart2spher(directions)[:, 1:] - torch.tensor([np.pi / 2, 0])

        # Relabel closest vertex to be heatmap peak
        dists = haversine_distances(self.sphereverts, directions)
        dists = self.sphereverts[np.argmin(dists, axis=0), :]

        # Calculate heatmap
        dists = haversine_distances(self.sphereverts, dists)
        dists_discrete = torch.min(torch.from_numpy(dists).float(), dim=1)[0]
        dists_discrete = torch.clamp(
            (torch.exp(self.alpha * (1 - dists_discrete)) - 1 / self.r) * (dists_discrete < self.r).long(),
            0,
            np.exp(self.alpha),
        )

        return dists_discrete

    def _get_sample(self, data: Dict[str, Any], index: float):
        data = move_data_to_device(data, self.device)

        # Global pointers
        id = data["id"]
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        scales = data["scales_meta_dict"]["scales"]
        nverts = data["scales_meta_dict"]["nverts"]
        centerline_function = data["centerline"]["interpolation"]

        # Local copies
        spheres = copy.deepcopy(data["scales"])

        # Calculate spheres and direction heatmap
        center, tangents = self._get_point(centerline_function, index)
        sampled_spheres = self._get_spheres(image, affine, spheres, center)
        direction = self._get_direction_heatmap(tangents)

        return {
            "global": {"id": id, "image": image, "affine": affine, "scales": scales, "nverts": nverts},
            "sample": {
                "spheres": sampled_spheres,
                "index": torch.tensor(index),
                "sampled_center": center,
                "center": center,
                "tangents": tangents,
                "direction": direction.reshape(-1, 1),
            },
        }

    def __call__(self, data: Dict[str, Any]):
        min_idx, max_idx = data["centerline"]["min_bound"], data["centerline"]["max_bound"]
        indexes = (max_idx - min_idx) * self.R.rand(self.num_samples) + min_idx
        data = [self._get_sample(data, index) for index in indexes]

        return data
