import os

from typing import Any, Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

from src.sire.utils.affine import get_rotation_matrix, transform_points


class VesselContour:
    def __init__(self, center: torch.Tensor, normal: torch.Tensor, points: Dict[str, Any]):
        self.center = center
        self.normal = normal
        self.points = points

    def _make_plane(self, size: int = 128):
        x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="xy")
        z = torch.zeros_like(x)
        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    def _estimate_scale(self):
        return max(
            [2 * torch.linalg.norm(points - self.center, axis=1).max().view(1) for _, points in self.points.items()]
        )

    def planar_projection(self, image: torch.Tensor, affine: torch.Tensor, size: int = 128):
        planar_points_dict = {}
        scale = self._estimate_scale()
        rot_matrix = get_rotation_matrix(self.normal.cpu()).to(self.center)

        # Compute planar contours
        for name, points in self.points.items():
            planar_points = (torch.linalg.inv(rot_matrix).float() @ (points.float() - self.center.float()).T).T
            spacing = size / (2 * scale)
            planar_points_dict[name] = planar_points[:, :-1] * spacing + (size / 2)

        # Compute planar image
        plane = self._make_plane(size)
        oriented_plane = transform_points(
            (rot_matrix @ (scale.to(self.center) * plane.to(self.center)).T).T + self.center,
            torch.linalg.inv(affine),
        )
        norm_oriented_plane = 2 * (oriented_plane / torch.tensor(image.shape)[[2, 1, 0]][None, ...].to(self.center)) - 1
        planar_img = (
            F.grid_sample(
                image[None, None, ...].float(),
                norm_oriented_plane[None, None, None, ...].float(),
                align_corners=False,
            )
            .squeeze()
            .reshape(size, size)
        )

        return planar_img, planar_points_dict, scale


class TrackedVessel:
    def __init__(self, image: torch.Tensor, affine: torch.Tensor, contours: List[VesselContour] = None):
        self.contours = [] if contours is None else self.contours
        self.edges = []
        self.image = image
        self.affine = affine

    def scrap_last(self):
        self.contours = self.contours[:-1]
        self.edges = self.edges[:-1]

    def update(self, contour: VesselContour):
        if len(self.contours) > 0:
            self.edges.append([len(self.contours) - 1, len(self.contours)])

        self.contours.append(contour)

    def build_centerline(self):
        centers = np.concatenate([contour.center.cpu().numpy() for contour in self.contours])
        edges = np.array(self.edges)

        flat_edges = np.c_[2 * np.ones(len(edges))[:, None], edges].flatten().astype(int)

        return pv.PolyData(centers, lines=flat_edges)

    def build_contours(self):
        point_dict = {}
        poly_contours = {}

        # Aggregate contours
        for contour in self.contours:
            for name, points in contour.points.items():
                if name not in point_dict.keys():
                    point_dict[name] = []

                point_dict[name].append(points)

        # Build polydata
        for name, points in point_dict.items():
            points = np.stack(points)
            num_contours, num_points, _ = points.shape

            contour_lines = np.array([[i, (i + 1) % num_points] for i in range(num_points)])
            all_contour_lines = np.concatenate([contour_lines + i * num_points for i in range(num_contours)])
            flat_lines = np.c_[2 * np.ones(len(all_contour_lines))[:, None], all_contour_lines].flatten().astype(int)

            poly_contour = pv.PolyData(points.reshape(-1, 3), lines=flat_lines)
            poly_contours[name] = poly_contour

        return poly_contours

    def save_planar_projections(self, output_dir: str):
        proper_contours = [contour for contour in self.contours if len(contour.points.keys()) != 0]

        for i, contour in enumerate(proper_contours):
            planar_img, planar_points_dict, scale = contour.planar_projection(self.image, self.affine)

            fig, ax = plt.subplots()
            ax.imshow(planar_img.cpu().numpy(), cmap="gray")
            ax.set_title(f"Max diameter: {(scale.item() / 2):.2f} mm")
            ax.axis("off")

            # Plot all the contours
            for j, (name, planar_points) in enumerate(planar_points_dict.items()):
                closed_points = torch.cat([planar_points, planar_points[0][None]])
                ax.plot(*closed_points.cpu().numpy().T, color=list(mcolors.TABLEAU_COLORS.values())[j], label=name)

            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"planar_contour_{i}.png"))
            plt.close(fig)

    def merge_at_start(self, tracked_vessel):
        contours = tracked_vessel.contours[1:][::-1]
        self.contours = contours + self.contours

        edges = [[i, i + 1] for i in range(len(self.contours) - 1)]
        self.edges = edges
