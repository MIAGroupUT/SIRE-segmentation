import json

from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
import typer

from scipy.ndimage import center_of_mass
from skimage.graph import MCP_Connect
from skimage.measure import regionprops

from src.sire.inference.totalsegmentator_processor import Roi, TotalSegmentatorProcessor
from src.sire.inference.utils.step_schedulers import ConstantStepScheduler
from src.sire.inference.utils.stopping_criterions import RoiStoppingCriterion, VesselDiameterStoppingCriterion
from src.sire.inference.utils.vessel_config import VesselConfig
from src.sire.utils.affine import transform_points

app = typer.Typer()


def load_manual_slicer_landmarks(filepath: str):
    with open(filepath) as f:
        points = json.load(f)["markups"][0]["controlPoints"]

    return {point["label"]: np.array(point["position"]) for point in points}


class AAATotalSegmentatorProcessor(TotalSegmentatorProcessor):
    def get_renal_seed(self, image: np.array, affine: torch.tensor, mask: np.array, renal: str, p: float = 0.5):
        def crop_renal(
            images: np.array,
            kidney_mask: np.array,
            aorta_mask: np.array,
            renal: str,
            padding: Tuple = (0, 0, 0, 0, 0, 0),
        ):
            min_x, min_y, min_z, max_x, max_y, max_z = regionprops(kidney_mask.astype(int))[0].bbox

            if renal == "right":
                max_z = regionprops(aorta_mask.astype(int))[0].bbox[5]
            else:
                min_z = regionprops(aorta_mask.astype(int))[0].bbox[2]

            cropped_images = images[
                :,
                min_x - padding[0] : max_x + padding[1],
                min_y - padding[2] : max_y + padding[3],
                min_z - padding[4] : max_z + padding[5],
            ]

            return cropped_images, np.array([min_x - padding[0], min_y - padding[2], min_z - padding[4]])

        # Crop for kidney and aorta
        labels = {"right": 2, "left": 3}
        kidney_mask = (mask == labels[renal]).astype(int)
        aorta_mask = (mask == 52).astype(int)
        bg_mask = (mask == 0).astype(int)

        cropped_images, offset = crop_renal(
            images=np.stack([image, kidney_mask, aorta_mask, bg_mask]),
            kidney_mask=kidney_mask,
            aorta_mask=aorta_mask,
            renal=renal,
            padding=(0, 40, 0, 0, 0, 0),
        )
        cropped_renal, cropped_kidney_mask, cropped_aorta_mask, cropped_bg_mask = cropped_images

        cropped_renal = cropped_renal.astype(float)
        cropped_kidney_mask = cropped_kidney_mask.astype(bool)
        cropped_aorta_mask = cropped_aorta_mask.astype(bool)
        cropped_bg_mask = cropped_bg_mask.astype(bool)

        # Connect center of masses for kidney and aorta with smallest cost path (renal)
        kidney_center = np.array(center_of_mass(cropped_kidney_mask)).reshape(-1, 3).astype(int)
        aorta_center = np.array(center_of_mass(cropped_aorta_mask)).reshape(-1, 3).astype(int)

        cost_array = 1 - ((cropped_renal - cropped_renal.min()) / cropped_renal.ptp())
        cost_array[~(cropped_bg_mask + cropped_aorta_mask + cropped_kidney_mask)] = 1

        mcp_connect = MCP_Connect(cost_array**10)
        mcp_connect.find_costs(kidney_center, ends=aorta_center, find_all_ends=False)

        trace = np.array(mcp_connect.traceback(aorta_center.reshape(3)))

        kidney_trace_mask = cropped_kidney_mask[trace[:, 0], trace[:, 1], trace[:, 2]] == 0
        aorta_trace_mask = cropped_aorta_mask[trace[:, 0], trace[:, 1], trace[:, 2]] == 0

        trace = trace[kidney_trace_mask & aorta_trace_mask]
        idx = np.percentile(np.arange(len(trace)), 100 * (1 - p)).astype(int)
        seed = trace[idx] + offset

        return transform_points(torch.tensor(seed[[2, 1, 0]].reshape(1, 3)), affine).numpy()

    def __call__(self, image_path: str, output_dir: str, seed_path: str = None, device: str = "cpu"):
        mask, affine = self.run(image_path, output_dir, device)
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

        # Setup rois
        segments = {
            # Outside of the heart, outside iliac arteries
            "abdominal_aorta": {
                "diameter": {"min": None, "max": None},
                "roi": Roi.intersection(mask, [Roi([51], "outside"), Roi([65, 66], "outside")]),
                "step": 1,
            },
            # Below aorta and outside of hips
            "iliac_left": {
                "diameter": {"min": None, "max": 50},
                "roi": Roi.intersection(
                    mask,
                    [Roi([77, 78], "outside"), Roi([52], "below", "min", dilate=20)],
                ),
                "step": 0.5,
            },
            "iliac_right": {
                "diameter": {"min": None, "max": 80},
                "roi": Roi.intersection(
                    mask,
                    [Roi([77, 78], "outside"), Roi([52], "below", "min", dilate=20)],
                ),
                "step": 0.5,
            },
            # Outside of dilated kidneys and eroted aorta
            "renal_right": {
                "diameter": {"min": 2, "max": 30},
                "roi": Roi.intersection(mask, [Roi([2, 3], "outside", dilate=-50), Roi([52], "outside", dilate=5)]),
                "step": 0.3,
            },
            "renal_left": {
                "diameter": {"min": 2, "max": 30},
                "roi": Roi.intersection(mask, [Roi([2, 3], "outside", dilate=-50), Roi([52], "outside", dilate=5)]),
                "step": 0.3,
            },
        }

        # Load seeds from file
        if seed_path is not None:
            seeds = load_manual_slicer_landmarks(seed_path)
            seed_segments = {}

            for name, point in seeds.items():
                if name in segments.keys():
                    seed_segments[name] = segments[name]
                    seed_segments[name]["seed"] = point

            segments = seed_segments

        # Else estimate
        else:
            segments["abdominal_aorta"]["seed"] = self._get_seed(
                mask, affine, segments["abdominal_aorta"]["roi"], 52, p=0.2
            )
            segments["iliac_right"]["seed"] = self._get_seed(mask, affine, segments["iliac_right"]["roi"], 66, p=0.8)
            segments["iliac_left"]["seed"] = self._get_seed(mask, affine, segments["iliac_left"]["roi"], 65, p=0.4)
            segments["renal_right"]["seed"] = self.get_renal_seed(image, affine, mask, "right", p=0.2)
            segments["renal_left"]["seed"] = self.get_renal_seed(image, affine, mask, "left", p=0.2)

        return [
            VesselConfig(
                name=name,
                seed_point=params["seed"],
                segment_every_n_steps=5,
                step_scheduler=ConstantStepScheduler(params["step"]),
                stopping_criterions=[
                    RoiStoppingCriterion(params["roi"]),
                    VesselDiameterStoppingCriterion(
                        max_diameter=params["diameter"]["max"], min_diameter=params["diameter"]["min"]
                    ),
                ],
            )
            for name, params in segments.items()
        ]
