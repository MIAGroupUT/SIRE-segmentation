from typing import Any

import numpy as np
import onnxruntime as ort
import pyvista as pv
import torch
import trimesh

from skimage import measure
from torch_cluster import knn
from tqdm import trange


class INRPointData:
    def __init__(
        self, points: torch.tensor, normals: torch.tensor = None, norm_scale: float = 1.8, device: str = "cpu"
    ):
        self.device = torch.device(device)

        self.scale, self.shift = self._normalize(points, norm_scale)
        self.points = (points.float().to(self.device) * self.scale) - self.shift

        if normals is None:
            self.normals = torch.zeros_like(self.points)
        else:
            self.normals = normals.float()

    def _normalize(self, points: torch.tensor, norm_scale: float):
        bbox_size = torch.max(points, dim=0)[0] - torch.min(points, dim=0)[0]
        scale = norm_scale / torch.max(bbox_size)

        points_scaled = points * scale
        shift = torch.mean(points_scaled, dim=0)
        shift[torch.argmax(bbox_size)] = torch.min(points_scaled[:, torch.argmax(bbox_size)]) + 0.5 * norm_scale

        return scale.float().to(self.device), shift.float().to(self.device)

    def _sample(self, n_points: int):
        perm = torch.randperm(self.points.shape[0])
        return self.points[perm[:n_points]], self.normals[perm[:n_points]]

    def _noise(self, n_points: int, var: float):
        return torch.randn(n_points, 3).to(self.points) * var

    def _component_distance(self, component: trimesh.Trimesh):
        component_points = torch.tensor(component.vertices).to(self.points)
        row, col = knn(component_points, self.points, 1)

        component_distance = torch.mean(torch.linalg.norm(self.points[row] - component_points[col], axis=0))

        return component_distance.item()

    def __call__(self, n_points: int, noise_var: int):
        coords_on, normals_on = self._sample(n_points)
        coords_off = self._sample(n_points)[0] + self._noise(n_points, noise_var)

        datadict = {
            "on": {"coords": coords_on, "normals": normals_on},
            "off": {"coords": coords_off},
        }

        return datadict

    def _to_world_coordinates(self, polydata: pv.PolyData):
        polydata.points = (polydata.points + self.shift.cpu().numpy()) / self.scale.cpu().numpy()
        return polydata

    def _extract_closest_component(self, polydata: pv.PolyData):
        faces_as_array = polydata.faces.reshape((polydata.n_faces_strict, 4))[:, 1:]
        tmesh = trimesh.Trimesh(polydata.points, faces_as_array)

        components = trimesh.graph.split(tmesh, only_watertight=False)

        if len(components) != 0:
            closest_idx = np.argmin([self._component_distance(component) for component in components])
            closest_component = components[closest_idx]

            flat_faces = (
                np.c_[3 * np.ones((len(closest_component.faces), 1)), closest_component.faces].flatten().astype(int)
            )
            polydata = pv.PolyData(closest_component.vertices, faces=flat_faces)

        return polydata

    def _sample_sdf(self, model_path: str, resolution: int, verbose: bool = False):
        inference_session = ort.InferenceSession(model_path)

        out_vol = np.zeros((resolution, resolution, resolution))
        zs = torch.linspace(-1.0, 1.0, steps=out_vol.shape[2])

        for z_it in trange(out_vol.shape[2], desc="Reconstructing mesh", leave=False, disable=not verbose):
            im_slice = np.zeros((resolution, resolution, 1))
            xs = torch.linspace(-1.0, 1.0, steps=im_slice.shape[1])
            ys = torch.linspace(-1.0, 1.0, steps=im_slice.shape[0])

            x, y = torch.meshgrid(xs, ys, indexing="xy")
            z = torch.ones_like(y) * zs[z_it]

            coords = torch.cat(
                [
                    x.reshape((np.prod(im_slice.shape[:2]), 1)),
                    y.reshape((np.prod(im_slice.shape[:2]), 1)),
                    z.reshape((np.prod(im_slice.shape[:2]), 1)),
                ],
                dim=1,
            )

            out = inference_session.run(None, {"input_coords": coords.cpu().numpy()})

            final_output = np.reshape(out[0], im_slice.shape[:2]).transpose()
            out_vol[:, :, z_it] = final_output

        return out_vol

    def reconstruct_pyvista(
        self, model_path: str, resolution: int = 256, extract_closest: bool = True, verbose: bool = True
    ) -> pv.PolyData:
        sdf = self._sample_sdf(model_path, resolution, verbose)

        verts, faces, _, _ = measure.marching_cubes(sdf, 0)
        verts = (2 * verts / (resolution - 1)) - 1

        flat_faces = np.c_[3 * np.ones((len(faces), 1)), faces].flatten().astype(int)
        mesh = pv.PolyData(verts, faces=flat_faces)
        mesh = mesh.compute_normals(auto_orient_normals=True)

        if extract_closest:
            mesh = self._extract_closest_component(mesh)

        return self._to_world_coordinates(mesh), sdf

    @classmethod
    def load_pyvista(cls, filename: str, clean_feature_edges: bool = False, device: Any = torch.device("cpu")):
        polydata = pv.read(filename)

        if clean_feature_edges:
            degenerated_polydata = polydata.extract_feature_edges(manifold_edges=False)

            x_points = torch.tensor(polydata.points, device=device)
            y_points = torch.tensor(degenerated_polydata.points, device=device)

            indices = knn(x_points, y_points, 1)

            polydata, _ = polydata.remove_points(indices[1].cpu())
            polydata = polydata.extract_largest()
            polydata = polydata.clean()

        points = torch.tensor(polydata.points)

        return cls(points, device=device)


class INRSkeletonPointData(INRPointData):
    def __init__(self, skeleton: torch.tensor, points: torch.tensor, norm_scale: float = 1.8, device: str = "cpu"):
        super().__init__(points, norm_scale=norm_scale, device=device)
        self.skeleton = (skeleton.float().to(self.device) * self.scale) - self.shift

    def __call__(self, n_points: int, noise_var: int):
        datadict = super().__call__(n_points, noise_var)
        datadict["skeleton"] = {"coords": self.skeleton}

        return datadict
