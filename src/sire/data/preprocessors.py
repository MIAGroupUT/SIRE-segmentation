from typing import Any, Dict, List

import numpy as np
import torch

from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh
from monai.config import KeysCollection
from monai.transforms import Compose, MapTransform, Transform
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

from src.sire.utils.affine import transform_points
from src.sire.utils.tracker import TrackerSphere


class BuildSIREScales(Transform):
    def __init__(
        self,
        scales: List[int],
        npoints: int = 32,
        subdivisions: int = 3,
    ):
        super().__init__()
        self.scales = scales
        self.npoints = npoints

        self.sphere = TrackerSphere(subdivisions=subdivisions)
        self.FaceToEdge = FaceToEdge(remove_faces=False)

    def __call__(self, data: Dict[str, Any]):
        affine = data["image_meta_dict"]["affine"]
        scales_coords = []

        for scale in self.scales:
            # Assume x-spacing is max. 0.1 cm, otherwise affine is in mms
            if np.abs(affine[0, 0]) < 0.1:
                scales_coords.append(torch.from_numpy(self.sphere.get_rays(self.npoints, scale / 10)).float())

            # Assume affine is in mm
            else:
                scales_coords.append(torch.from_numpy(self.sphere.get_rays(self.npoints, scale)).float())

        scales_coords = torch.cat(scales_coords, dim=0)

        origin = affine[:-1, -1]
        scales_coords += origin

        scales_coords = transform_points(scales_coords, torch.linalg.inv(affine))
        pos, facestack = self._structure_faces()

        data["scales"] = Data(coords=scales_coords, face=facestack, pos=pos)
        data["scales"] = self.FaceToEdge(data["scales"])
        data["scales_meta_dict"] = {
            "scales": torch.tensor(self.scales),
            "nverts": torch.tensor(len(self.sphere.sphere.vertices)),
        }

        return data

    def _structure_faces(self):
        faces = torch.from_numpy(self.sphere.sphere.faces.T).long()
        pos = torch.from_numpy(self.sphere.sphere.vertices).float()
        facestack = torch.hstack([faces + i * pos.shape[0] for i in range(len(self.scales))])
        pos = torch.vstack([pos] * len(self.scales))
        return pos, facestack


class BuildForGEMGCN(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

        self.transform = Compose([compute_normals_edges_from_mesh, SimpleGeometry(), GemPrecomp(2, 2)])

    def __call__(self, data: Dict[str, Any]):
        d = dict(data)
        for key in self.key_iterator(d):
            sample = d[key]
            sample = self.transform(sample)

            d[key] = sample

        return d
