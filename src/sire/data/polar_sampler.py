from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.sire.utils.affine import get_rotation_matrix, transform_points


class PolarSampler:
    def __init__(self, n_rad: int = 128, n_angles: int = 64):
        self.n_rad = n_rad
        self.n_angles = n_angles

    def _make_disk(self, diameter: int = 2, padding: int = 0):  # 2 cm diameter, 1 cm radius, centered at [0,0]
        r = torch.linspace(0, diameter, self.n_rad + 2 * padding) - diameter / 2
        theta = torch.linspace(0, 2 * np.pi, self.n_angles + 1)[: self.n_angles]
        R, Theta = torch.meshgrid(r, theta, indexing="xy")
        x = R.flatten() * torch.cos(Theta.flatten())
        y = R.flatten() * torch.sin(Theta.flatten())
        z = torch.zeros_like(x)
        return torch.stack([x, y, z], dim=-1)

    def _get_radii(
        self,
        center: torch.tensor,
        points: torch.tensor,
        scale: torch.tensor,
        rot_matrix: torch.tensor,
        padding: int = 0,
    ):
        lumen_cont = (torch.linalg.inv(rot_matrix).float() @ (points.float() - center.float()).T).T
        r = torch.sqrt(lumen_cont[:, 0] ** 2 + lumen_cont[:, 1] ** 2)
        theta = torch.arctan2(lumen_cont[:, 1], lumen_cont[:, 0])
        radii = torch.from_numpy(
            np.interp(
                np.linspace(0, 2 * np.pi, self.n_angles + 1)[:-1],
                theta.cpu().numpy(),
                r.cpu().numpy(),
                period=np.pi * 2,
            )
        ).to(center)
        spacing = (self.n_rad + 2 * padding) / (2 * scale)
        return radii * spacing

    def inverse(
        self,
        polar_radii_batch: torch.tensor,
        center_batch: torch.tensor,
        normal_batch: torch.tensor,
        scale_batch: torch.tensor,
        padding: int = 0,
    ):
        rot_coords_batch = []

        for polar_radii_all, center, normal, scale in zip(polar_radii_batch, center_batch, normal_batch, scale_batch):
            _, num_radii, _ = polar_radii_all.shape
            all_rot_coords = []

            # Convert to contours for all radii:
            for i in range(num_radii):
                polar_radii = polar_radii_all[:, i].flatten()

                spacing = (self.n_rad + 2 * padding) / (2 * scale)
                polar_radii = polar_radii / spacing

                theta = torch.linspace(0, 2 * np.pi, self.n_angles + 1)[: self.n_angles].to(polar_radii)
                x_coords, y_coords = polar_radii * torch.cos(theta), polar_radii * torch.sin(theta)
                z_coords = torch.zeros_like(x_coords)

                coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)

                rot_matrix = get_rotation_matrix(normal.cpu()).to(coords)
                rot_coords = (rot_matrix @ coords.T).T + center
                all_rot_coords.append(rot_coords)

            rot_coords_batch.append(torch.stack(all_rot_coords, dim=1))

        return torch.stack(rot_coords_batch)

    def __call__(self, data: Dict[str, Any], scales: torch.tensor, padding: int = 0):
        image = data["global"]["image"][0]
        affine = data["global"]["affine"][0]

        center_batch = data["sample"]["center"]
        normal_batch = data["sample"]["normal"]
        contours_batch = data["sample"]["contours"]

        polar_images, polar_contours, polar_masks = [], [], []

        # Iterate over batches
        for center, normal, contours, scale in zip(center_batch, normal_batch, contours_batch, scales):
            _, num_contours, _ = contours.shape
            rot_matrix = get_rotation_matrix(normal.cpu()).to(center)
            all_radii = []

            # Compute radii for all contours
            for i in range(num_contours):
                radii = self._get_radii(center, contours[:, i], scale, rot_matrix, padding)[..., None]
                all_radii.append(radii)

            radii = torch.stack(all_radii, 1)

            # Compute polar image
            disk = self._make_disk(padding=padding)
            oriented_disk = transform_points(
                (rot_matrix @ (scale.to(center) * disk.to(center)).T).T + center, torch.linalg.inv(affine)
            )
            norm_oriented_disk = 2 * (oriented_disk / torch.tensor(image.shape)[[2, 1, 0]][None, ...].to(center)) - 1
            polar_grid = norm_oriented_disk.reshape(self.n_angles, self.n_rad + 2 * padding, 3).float()
            polar_img = F.grid_sample(
                image[None, None, ...], polar_grid[None, None, ...], align_corners=False
            ).squeeze()
            polar_img = F.pad(polar_img[None, None], (0, 0, padding, padding), "circular").squeeze()

            # Compute mask for segmentation task
            polar_mask = torch.zeros((self.n_angles, self.n_rad)).to(polar_img)

            for i in range(self.n_angles):
                upper_bound = torch.arange(self.n_rad).to(radii) > self.n_rad / 2
                lower_bound = torch.arange(self.n_rad).to(radii) < radii[i] + self.n_rad / 2
                polar_mask[i] = lower_bound & upper_bound

            # Collect results
            polar_images.append(polar_img.float())
            polar_contours.append(radii.float())
            polar_masks.append(polar_mask.float())

        return torch.stack(polar_images), torch.stack(polar_contours), torch.stack(polar_masks)
