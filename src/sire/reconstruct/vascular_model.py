import glob
import os

from typing import Dict, List

import numpy as np
import pyvista as pv
import torch

from scipy.spatial.distance import cdist
from torch_cluster import knn


class VascularModel:
    """Vascular model object for centerline correction and pruning"""

    def __init__(self, contours: Dict[str, np.array], npoints: int = 128):
        self.contours = {name: contour.reshape(-1, npoints, 3) for name, contour in contours.items()}
        self.centerlines = {name: contour.mean(axis=1) for name, contour in self.contours.items()}
        self.correct_dict = {
            name: np.ones(len(centerline), dtype=bool) for name, centerline in self.centerlines.items()
        }

    def get_corrected_contour(self, branch: str, connect_to: np.array = None):
        contours = self.contours[branch][self.correct_dict[branch]]
        centerline = contours.mean(axis=1)

        if connect_to is not None:
            if np.linalg.norm(vec1 := (connect_to - centerline[-1])) > np.linalg.norm(
                vec0 := (connect_to - centerline[0])
            ):
                contours = np.concatenate([(contours[0] + vec0).reshape(1, -1, 3), contours])
            else:
                contours = np.concatenate([contours, (contours[-1] + vec1).reshape(1, -1, 3)])

        return contours

    def _get_merge_point(self, main_centerline: np.array, segment_centerline: np.array, tol: float):
        main_points = torch.tensor(main_centerline)
        segment_points = torch.tensor(segment_centerline)

        _, row = knn(main_points, segment_points, 1)
        distance = torch.linalg.norm(segment_points - main_points[row], dim=1)
        merge_mask = (distance < tol).int()

        if distance[0] > distance[-1]:
            if merge_mask.sum() == 0:
                threshold = len(merge_mask) - 1
            else:
                threshold = torch.argwhere(merge_mask).min()

            merge_mask = np.zeros_like(merge_mask)
            merge_mask[threshold:] = 1
        else:
            if merge_mask.sum() == 0:
                threshold = 0
            else:
                threshold = torch.argwhere(merge_mask).max()

            merge_mask = np.zeros_like(merge_mask)
            merge_mask[:threshold] = 1

        return segment_points[threshold].numpy(), ~merge_mask.astype(bool)

    def correct_bifurcation(self, main_branch: str, side_branch_1: str, side_branch_2: str, merge_tol: float):
        def get_bifurcation_point(branch: np.array, anchor1: np.array, anchor2: np.array):
            top_point = branch[np.argmax(branch[:, -1])]

            # Specify higher point to be bifurcation point
            if np.linalg.norm(top_point - anchor1) > np.linalg.norm(top_point - anchor2):
                anchor = anchor2
            else:
                anchor = anchor1

            return anchor

        def trim_by_bifurcation(branch: np.array, bifurcation_point: np.array, z_mode: str):
            index = np.argmin(np.linalg.norm(branch - bifurcation_point, axis=1))
            point_mask = np.zeros(len(branch))

            if index > 0 and index < len(branch) - 1:
                if branch[0, -1] > branch[-1, -1]:
                    if z_mode == "min":
                        point_mask[index:] = 1
                    else:
                        point_mask[: index + 1] = 1
                else:
                    if z_mode == "min":
                        point_mask[: index + 1] = 1
                    else:
                        point_mask[index:] = 1

                return point_mask.astype(bool)

            else:
                return ~(point_mask.astype(bool))

        anchor_1, _ = self._get_merge_point(self.centerlines[main_branch], self.centerlines[side_branch_1], merge_tol)
        anchor_2, _ = self._get_merge_point(self.centerlines[main_branch], self.centerlines[side_branch_2], merge_tol)
        bifurcation_point = get_bifurcation_point(self.centerlines[main_branch], anchor_1, anchor_2)

        self.correct_dict[side_branch_1] = trim_by_bifurcation(
            self.centerlines[side_branch_1], bifurcation_point, "min"
        )
        self.correct_dict[side_branch_2] = trim_by_bifurcation(
            self.centerlines[side_branch_2], bifurcation_point, "min"
        )
        self.correct_dict[main_branch] = trim_by_bifurcation(self.centerlines[main_branch], bifurcation_point, "max")

        endings = self.centerlines[main_branch][self.correct_dict[main_branch]][[0, -1]]
        main_anchor = np.argmin(np.linalg.norm(endings - bifurcation_point, axis=1))

        return endings[main_anchor]

    def correct_overlap(self, main_branch: str, side_branch: str, merge_tol: float):
        _, point_mask = self._get_merge_point(self.centerlines[main_branch], self.centerlines[side_branch], merge_tol)
        self.correct_dict[side_branch] = point_mask.astype(bool)

    def get_pruned_contour(
        self,
        branch: str,
        ref_branch: str,
        distance: float,
        connect_centerline: bool = True,
        keep_overlap: bool = True,
    ):
        branch_centerline = self.centerlines[branch][self.correct_dict[branch]]
        ref_branch_centerline = self.centerlines[ref_branch][self.correct_dict[ref_branch]]

        # Determine direction of pruning
        all_distances = cdist(branch_centerline[[0, -1]], ref_branch_centerline)
        indices = np.argmin(all_distances, axis=1)
        distances = np.min(all_distances, axis=1)

        # Prune on determined side to given distance
        if distances[0] > distances[1]:
            cumdist = np.cumsum(np.linalg.norm(np.diff(branch_centerline, axis=0), axis=1))
            ncontours = np.argwhere(cumdist < 10 * distance).max()

            # Prune and include overlap
            if keep_overlap:
                pruned_contour = np.concatenate(
                    [
                        self.contours[branch][self.correct_dict[branch]][-ncontours:],
                        self.contours[branch][~self.correct_dict[branch]],
                    ]
                )
            else:
                pruned_contour = self.contours[branch][self.correct_dict[branch]][-ncontours:]

            # Connect to ref branch if necessary
            pruned_centerline = pruned_contour.mean(axis=1)
            if connect_centerline:
                root = pruned_centerline[-1]

                tangent = ref_branch_centerline[indices[1]] - root
                unit_tangent = tangent / np.linalg.norm(tangent)

                spacing = np.mean(np.linalg.norm(np.diff(branch_centerline, axis=0), axis=1))
                nrings = int(np.linalg.norm(tangent) / spacing)

                for _ in range(nrings - 1):
                    vec = spacing * unit_tangent
                    pruned_centerline = np.concatenate([pruned_centerline, (pruned_centerline[-1] + vec)[None]])

        else:
            cumdist = np.cumsum(np.linalg.norm(np.diff(branch_centerline[::-1], axis=0), axis=1))
            ncontours = np.argwhere(cumdist < 10 * distance).max()

            # Prune and include overlap
            if keep_overlap:
                pruned_contour = np.concatenate(
                    [
                        self.contours[branch][~self.correct_dict[branch]],
                        self.contours[branch][self.correct_dict[branch]][:ncontours],
                    ]
                )
            else:
                pruned_contour = self.contours[branch][self.correct_dict[branch]][:ncontours]

            # Connect to ref branch if necessary
            pruned_centerline = pruned_contour.mean(axis=1)
            if connect_centerline:
                root = pruned_centerline[0]

                tangent = ref_branch_centerline[indices[0]] - root
                unit_tangent = tangent / np.linalg.norm(tangent)

                spacing = np.mean(np.linalg.norm(np.diff(branch_centerline, axis=0), axis=1)) / 2
                nrings = int(np.linalg.norm(tangent) / spacing)

                for _ in range(nrings - 1):
                    vec = spacing * unit_tangent
                    pruned_centerline = np.concatenate([(pruned_centerline[0] + vec)[None], pruned_centerline])

        return pruned_contour, pruned_centerline

    @classmethod
    def load_from_directory(
        cls,
        root_dir: str,
        filenames: List[str],
        extension: str = ".vtp",
        npoints: int = 128,
    ):
        contours = {
            filename: pv.read(glob.glob(os.path.join(root_dir, f"*{filename}*{extension}"))[0]).points
            for filename in filenames
        }
        return cls(contours, npoints)
