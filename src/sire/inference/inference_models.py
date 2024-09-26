import os

from typing import Any, Dict, List

import torch

from monai.transforms import Compose
from pytorch_lightning.utilities import move_data_to_device
from torch_geometric.loader import DataLoader

from src.sire.data.preprocessors import BuildForGEMGCN, BuildSIREScales
from src.sire.data.transforms import SampleSIRE
from src.sire.models.sire_base import SIREBase
from src.sire.models.sire_seg import SIRESegmentation
from src.sire.models.sire_tracker import SIRETracker
from src.sire.utils.wandb import load_wandb_artifact


class InferenceModelBase:
    """Base class for SIRE models inference wrappers"""

    def __init__(
        self,
        model: SIREBase,
        scales: List[int],
        npoints: int = 32,
        subdivisions: int = 3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.transforms = Compose(
            [
                BuildSIREScales(
                    scales=scales,
                    npoints=npoints,
                    subdivisions=subdivisions,
                ),
                BuildForGEMGCN(keys=["scales"]),
            ]
        )
        self.sampler = SampleSIRE(npoints=npoints, subdivisions=subdivisions, device=device)
        self.device = device

    def transform_input(self, data):
        return self.transforms(data)

    def __call__(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def load_from_wandb(
        cls,
        model_class: SIREBase,
        scales: List[int],
        npoints: int,
        subdivisions: int,
        names: List[str],
        wandb_config: Dict[str, Any],
        device: str = "cpu",
    ):
        artifact_dir = load_wandb_artifact(**wandb_config)
        model = model_class.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))

        return cls(model, scales, npoints, subdivisions, names, device)


class TrackerInferenceModel(InferenceModelBase):
    """SIRE tracker model inference wrapper

    Args:
        model (SIRETracker): loaded SIRE tracker model
        scales (List[int]): list of scales to use for SIRE backbone
        npoints (int, optional): number of points along rays in SIRE backbone. Defaults to 32.
        subdivisions (int, optional): resolution of spheres in SIRE backbone. Defaults to 3.
        device (str, optional): device to run the model on. Defaults to "cpu".
    """

    def __call__(self, data, point):
        data = self.sampler(data, point)
        data = list(DataLoader([data]))[0]

        with torch.no_grad():
            output = self.model(data)
            output = move_data_to_device(output, "cpu")

        return output

    @classmethod
    def load_from_wandb(
        cls,
        scales: List[int],
        npoints: int,
        subdivisions: int,
        wandb_config: Dict[str, Any],
        device: str = "cpu",
    ):
        artifact_dir = load_wandb_artifact(**wandb_config)
        model = SIRETracker.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))

        return cls(model, scales, npoints, subdivisions, device)


class SegmentationInferenceModel(InferenceModelBase):
    """SIRE segmentetation model inference wrapper

    Args:
        model (SIRESegmentation): loaded SIRE segmentation model
        names (List[str]): list of contour names that the model outputs shoulde be stored at,
                           provide name for each channel
        scales (List[int]): list of scales to use for SIRE backbone
        npoints (int, optional): number of points along rays in SIRE backbone. Defaults to 32.
        subdivisions (int, optional): resolution of spheres in SIRE backbone. Defaults to 3.
        device (str, optional): device to run the model on. Defaults to "cpu".
    """

    def __init__(
        self,
        model: SIRESegmentation,
        names: List[str],
        scales: List[int],
        npoints: int = 32,
        subdivisions: int = 3,
        device: str = "cpu",
    ):
        super().__init__(model, scales, npoints, subdivisions, device)
        self.names = names

    def __call__(self, data, point, direction):
        data = self.sampler(data, point)
        data = list(DataLoader([data]))[0]
        data["sample"]["normal"] = direction.view(-1, 3)

        with torch.no_grad():
            output = self.model(data)
            output = move_data_to_device(output, "cpu")

        return output

    @classmethod
    def load_from_wandb(
        cls,
        names: List[str],
        scales: List[int],
        npoints: int,
        subdivisions: int,
        wandb_config: Dict[str, Any],
        device: str = "cpu",
    ):
        artifact_dir = load_wandb_artifact(**wandb_config)
        model = SIRESegmentation.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))

        return cls(model, names, scales, npoints, subdivisions, device)
