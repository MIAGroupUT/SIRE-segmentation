from typing import List

import numpy as np
import pyvista as pv

from pysdf import SDF
from skimage import measure
from tqdm.auto import tqdm


class MeshSDFMerger:
    """Merges given list of meshes through SDF sum"""

    def _get_global_grid(self, mesh_list: List[pv.PolyData], voxel_size: int, margin: int):
        all_points = np.concatenate([mesh.points for mesh in mesh_list])

        x_min, y_min, z_min = all_points.min(axis=0)
        x_max, y_max, z_max = all_points.max(axis=0)

        x_steps = np.round((x_max - x_min + 2 * margin) / voxel_size)
        y_steps = np.round((y_max - y_min + 2 * margin) / voxel_size)
        z_steps = np.round((z_max - z_min + 2 * margin) / voxel_size)

        xs = np.linspace(x_min - margin, x_max + margin, num=int(x_steps))
        ys = np.linspace(y_min - margin, y_max + margin, num=int(y_steps))
        zs = np.linspace(z_min - margin, z_max + margin, num=int(z_steps))

        trans = np.array([x_min, y_min, z_min]) - margin
        scale = np.array([xs[-1] - xs[0], ys[-1] - ys[0], zs[-1] - zs[0]]) / (np.array([x_steps, y_steps, z_steps]) - 1)

        return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(len(xs), len(ys), len(zs), 3), trans, scale

    def _get_grid_mask(self, grid: np.array, mesh: pv.PolyData, margin: int):
        x_min, y_min, z_min = mesh.points.min(axis=0)
        x_max, y_max, z_max = mesh.points.max(axis=0)

        vol_mask = np.zeros(grid.shape[:-1])
        x_mask = (x_min - margin <= grid[..., 0]) & (grid[..., 0] <= x_max + margin)
        y_mask = (y_min - margin <= grid[..., 1]) & (grid[..., 1] <= y_max + margin)
        z_mask = (z_min - margin <= grid[..., 2]) & (grid[..., 2] <= z_max + margin)

        vol_mask[x_mask & y_mask & z_mask] = 1

        return vol_mask.astype(bool)

    def run(self, mesh_list: pv.PolyData, voxel_size: int, margin: int = 2, k: int = 2, verbose: bool = False):
        grid, trans, scale = self._get_global_grid(mesh_list, voxel_size, margin)

        for i, mesh in tqdm(enumerate(mesh_list), total=len(mesh_list), disable=not verbose):
            mesh = mesh.clean().triangulate().connectivity("largest").extract_geometry()
            faces = mesh.faces.reshape(-1, 4)[:, 1:]

            # Sample whole SDF for first sample
            if i == 0:
                points = grid.reshape(-1, 3)

                sdf = SDF(mesh.points, faces)
                points_sdf = -sdf(points)

                vol = points_sdf.reshape(grid.shape[:-1])

            # For other samples perform smooth union for ROI
            else:
                grid_mask = self._get_grid_mask(grid, mesh, margin)
                points = grid[grid_mask]

                sdf = SDF(mesh.points, faces)
                points_sdf = -sdf(points)

                points_vol = vol[grid_mask]
                points_smoothed = -k * np.log(np.exp(-points_sdf / k) + np.exp(-points_vol / k))

                vol[grid_mask] = points_smoothed

        verts, faces, _, _ = measure.marching_cubes(vol, 0, spacing=scale)
        verts += trans

        flat_faces = np.c_[3 * np.ones((len(faces), 1)), faces].flatten().astype(int)
        mesh = pv.PolyData(verts, faces=flat_faces)
        mesh = mesh.smooth_taubin(normalize_coordinates=True)

        return mesh
