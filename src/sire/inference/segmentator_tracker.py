import logging
import os

from dataclasses import asdict
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk
import torch

from logdecorator import log_on_end, log_on_start
from monai.transforms import Compose, ScaleIntensityRanged
from sklearn.metrics.pairwise import haversine_distances
from tqdm.auto import tqdm

from src.sire.inference.inference_models import SegmentationInferenceModel, TrackerInferenceModel
from src.sire.inference.utils.step_schedulers import ConstantStepScheduler, StepSchedulerBase
from src.sire.inference.utils.stopping_criterions import AlreadyTrackedStoppingCriterion, StoppingCriterionBase
from src.sire.inference.utils.tracked_vessel import TrackedVessel, VesselContour
from src.sire.inference.utils.vessel_config import VesselConfig
from src.sire.utils.affine import cart2spher, get_affine

RICH_INFO = 25
logging._levelToName[RICH_INFO] = "RICH_INFO"
logging._nameToLevel["RICH_INFO"] = RICH_INFO


class SegmentatorTrackerPipeline:
    """Tracking and contour regression pipeline for SIRE models.

    Args:
        tracker_model (TrackerInferenceModel): loaded tracker model
        segmentation_models (List[SegmentationInferenceModel]): list of loaded segmentation models
        logging_level (int, optional): at which level the logging should be perform, use default RICH_INFO
                                       to see how the segments are tracked, from which points, where they stop etc.
                                       Defaults to RICH_INFO.
    """

    def __init__(
        self,
        tracker_model: TrackerInferenceModel,
        segmentation_models: List[SegmentationInferenceModel],
        logging_level: int = RICH_INFO,
    ):
        self.logging_level = logging_level
        logging.basicConfig(level=logging_level, format="%(levelname)s (%(asctime)s): %(message)s")

        self.tracker_model = tracker_model
        self.segmentation_models = segmentation_models

        self.pre_transforms = Compose(
            [ScaleIntensityRanged(keys=["image"], a_min=-400, a_max=800, b_min=0, b_max=1, clip=False)]
        )

        self.last_n_directions = []

    @log_on_start(RICH_INFO, "Preparing data")
    def _prepare_data(self, image: np.array, affine: np.array, seed_point: np.array):
        """Convert image data to the input dictionary."""
        data = {
            "id": torch.tensor([0]),
            "image": torch.from_numpy(image),
            "image_meta_dict": {"affine": affine},
        }

        data = self.pre_transforms(data)
        data = {
            "tracking": self.tracker_model.transform_input(data),
            "segmentation": [
                segmentation_model.transform_input(data) for segmentation_model in self.segmentation_models
            ],
        }

        return (
            data,
            torch.from_numpy(seed_point).reshape(3),
        )

    def _get_direction(self, heatmap: torch.tensor, prev_direction: torch.tensor):
        """Get tracking direction from the heatmap - with previous direction masking."""
        prev_direction /= torch.linalg.norm(prev_direction)
        prev_dir_spher = cart2spher(prev_direction.reshape(1, -1))[:, 1:] - np.array([np.pi / 2, 0])

        dists = torch.from_numpy(haversine_distances(self.tracker_model.sampler.sphereverts, prev_dir_spher))
        heatmap[:, dists > np.pi / 3] = 0
        ind = torch.argmax(heatmap, dim=1)

        return torch.from_numpy(self.tracker_model.sampler.sphere.cartverts[ind, :])

    def _init_tracking(self, data: Dict[str, Any], point: torch.tensor):
        """Initilize tracking at the point by finding two-side directions."""

        # Call model
        heatmap = self.tracker_model(data["tracking"], point)

        # Get leading direction
        ind_1 = torch.argmax(heatmap, dim=1)
        direction_spher = self.tracker_model.sampler.sphereverts[ind_1, :].view(-1, 2)

        # Get opposite direction
        dists = torch.from_numpy(haversine_distances(self.tracker_model.sampler.sphereverts, direction_spher))
        heatmap[:, dists < np.pi / 2] = 0
        ind_2 = torch.argmax(heatmap, dim=1)

        directions = torch.from_numpy(
            np.stack([self.tracker_model.sampler.sphere.cartverts[ind, :] for ind in [ind_1, ind_2]])
        )

        return directions

    def _stop_tracking(
        self,
        stopping_criterions: List[StoppingCriterionBase],
        iteration: int,
        point: torch.tensor,
        data: Dict[str, Any],
        vessel_contour: VesselContour,
    ) -> bool:
        """Evaluates list of stopping criterions to check whether tracking should terminate."""
        return [
            stopping_criterion(iteration=iteration, point=point, data=data["tracking"], vessel_contour=vessel_contour)
            for stopping_criterion in stopping_criterions
        ]

    @log_on_end(RICH_INFO, "Stopped after {iteration!r} iterations: {result!r}")
    def _check_triggered_criterion(
        self, stopping_criterions: List[StoppingCriterionBase], criterions_state: List[bool], iteration: int
    ):
        """Checks which stopping criterion triggered - for logging sake."""
        return [
            str(stopping_criterion)
            for stopping_criterion, state in zip(stopping_criterions, criterions_state)
            if state is True
        ]

    def _iteration(
        self,
        iteration: int,
        data: Dict[str, Any],
        point: torch.tensor,
        prev_direction: torch.tensor,
        segment_every_n_steps: int,
        average_last_n_directions: int = 5,
    ) -> VesselContour:
        """Run single tracker iteration - get contour and direction at the given point."""

        # Call model
        heatmap = self.tracker_model(data["tracking"], point)
        direction = self._get_direction(heatmap, prev_direction)

        # Collect last n-directions
        self.last_n_directions.append(direction)

        if len(self.last_n_directions) > average_last_n_directions:
            self.last_n_directions = self.last_n_directions[1:]

        # Consider directions as mean of last n directions
        direction = torch.stack(self.last_n_directions).mean(dim=0)
        direction /= torch.linalg.norm(direction)

        self.last_n_directions[-1] = direction
        contours = {}

        # Run segmentation for each provided model seperately (if provided and n-step)
        if self.segmentation_models is not None and iteration % segment_every_n_steps == 0:
            for segmentation_model, data in zip(self.segmentation_models, data["segmentation"]):
                _, polar_contour, _, scale = segmentation_model(data, point, direction)
                padding = segmentation_model.model.head.padding

                # Split channels to the
                for i, contour_name in enumerate(segmentation_model.names):
                    cartesian_contour = segmentation_model.model.polar_sampler.inverse(
                        polar_contour[:, :, i].unsqueeze(2),
                        point.view(-1, 3),
                        direction.view(-1, 3),
                        scale,
                        padding=padding,
                    ).view(-1, 3)

                    contours[contour_name] = cartesian_contour

        return VesselContour(point, direction, contours)

    @log_on_start(RICH_INFO, "Direction: {direction!r}")
    def _track_direction(
        self,
        data: Dict[str, Any],
        seed_point: torch.tensor,
        direction: torch.tensor,
        segment_every_n_steps: int,
        step_scheduler: StepSchedulerBase,
        stopping_criterions: List[StoppingCriterionBase],
    ) -> TrackedVessel:
        iteration = 0
        self.last_n_directions = []

        # Init tracked vessel
        tracked_vessel = TrackedVessel(data["tracking"]["image"], data["tracking"]["image_meta_dict"]["affine"])
        vessel_contour = VesselContour(seed_point, direction, None)
        point = seed_point

        # Run tracking and segmentation until any of conditions is triggered (at least one iteration)
        while iteration == 0 or not np.any(
            criterions_state := self._stop_tracking(stopping_criterions, iteration, point, data, vessel_contour)
        ):
            vessel_contour = self._iteration(iteration, data, point, vessel_contour.normal, segment_every_n_steps)

            point = vessel_contour.center + vessel_contour.normal * step_scheduler()
            tracked_vessel.update(vessel_contour)

            iteration += 1

        self._check_triggered_criterion(stopping_criterions, criterions_state, iteration)

        # Scrap last if tracking succeeded - lasted more than one iteration
        if iteration > 1:
            tracked_vessel.scrap_last()

        return tracked_vessel

    @log_on_start(RICH_INFO, "Seed_point: {seed_point!r}")
    def _track_from_point(
        self,
        data: Dict[str, Any],
        seed_point: torch.tensor,
        segment_every_n_steps: int,
        step_scheduler: StepSchedulerBase,
        stopping_criterions: List[StoppingCriterionBase],
    ) -> TrackedVessel:
        """Perform tracking from the given seed point - both directions."""
        directions = self._init_tracking(data, seed_point)

        assert len(directions) == 2

        forward_track = self._track_direction(
            data, seed_point, directions[0], segment_every_n_steps, step_scheduler, stopping_criterions
        )
        backward_track = self._track_direction(
            data, seed_point, directions[1], segment_every_n_steps, step_scheduler, stopping_criterions
        )
        forward_track.merge_at_start(backward_track)

        return forward_track

    @log_on_end(RICH_INFO, "Finished")
    def run_single(
        self,
        image: np.array,
        affine: np.array,
        seed_point: np.array,
        segment_every_n_steps: int = 1,
        step_scheduler: StepSchedulerBase = ConstantStepScheduler(1),
        stopping_criterions: Tuple[StoppingCriterionBase] = (),
        **kwargs,
    ) -> TrackedVessel:
        """Run pipeline for single segment.

        Args:
            image (np.array): loaded image in numpy
            affine (np.array): loaded image affine matrice in numpy
            seed_point (np.array): 3D seed point where the tracking should start
            segment_every_n_steps (int, optional): every how many steps should the contour be delineated. Defaults to 1.
            step_scheduler (StepSchedulerBase, optional): how to schedule tracker step size. Defaults to ConstantStepScheduler(1).
            stopping_criterions (Tuple[StoppingCriterionBase], optional): stopping criterions for the tracking. Defaults to ().

        Returns:
            TrackedVessel: tracked vessel segment
        """

        data, seed_point = self._prepare_data(image, affine, seed_point)
        tracked_vessel = self._track_from_point(
            data,
            seed_point,
            segment_every_n_steps,
            step_scheduler,
            stopping_criterions,
        )

        return tracked_vessel

    @log_on_start(RICH_INFO, "Pipeline started")
    @log_on_end(RICH_INFO, "Pipeline finished")
    def run(
        self,
        image: Union[str, np.array],
        affine: np.array = None,
        vessel_configs: List[VesselConfig] = None,
        already_tracked_distance: float = 0,
        output_dir: str = None,
    ) -> Dict[str, Dict[str, pv.PolyData]]:
        """Run pipeline for multiple segments on one image.
        Single segments are provided as VesselConfig objects.

        Args:
            image (Union[str, np.array]): loaded image in numpy or path to load an image from
            affine (np.array, optional): if image provided in numpy, the affine matrice needs to be provided as well. Defaults to None.
            vessel_configs (List[VesselConfig], optional): single segment configurations for tracking. Defaults to None.
            already_tracked_distance (float, optional): specify whether tracking should stop
                                                        if it was already tracked by other segment. Defaults to 0.
            output_dir (str, optional): path to output directory, if None then not saving. Defaults to None.

        Returns:
            Dict[str, Dict[str, pv.PolyData]]: dictionary of tracked centerlines and contours for all provided segments
        """
        # Load from path if str given
        if isinstance(image, str):
            itk_image = sitk.ReadImage(image)
            affine, _ = get_affine(itk_image)
            image = sitk.GetArrayFromImage(itk_image)

        # Save seed point if output dir given
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            points = [vc.seed_point.reshape(3) for vc in vessel_configs]
            names = [vc.name for vc in vessel_configs]

            pd.DataFrame.from_records(points, index=names).to_csv(os.path.join(output_dir, "seeds.csv"))

        # Run pipeline for each vessel configuration provided
        results = {}
        already_tracked_stopping = AlreadyTrackedStoppingCriterion(None, already_tracked_distance)

        with tqdm(total=len(vessel_configs), disable=self.logging_level > RICH_INFO) as pbar:
            for vessel_config in vessel_configs:
                name = vessel_config.name

                pbar.set_description(name)

                vessel_config.stopping_criterions.append(already_tracked_stopping)
                tracked_vessel: TrackedVessel = self.run_single(image=image, affine=affine, **asdict(vessel_config))
                results[name] = {
                    "centerline": tracked_vessel.build_centerline(),
                    "contour": tracked_vessel.build_contours(),
                }

                if output_dir is not None:
                    # Save centerline
                    os.makedirs(os.path.join(output_dir, "centerline"), exist_ok=True)
                    results[name]["centerline"].save(os.path.join(output_dir, "centerline", f"centerline_{name}.vtp"))

                    # Save contours
                    for contour_name, contour in results[name]["contour"].items():
                        os.makedirs(os.path.join(output_dir, "contour", contour_name), exist_ok=True)
                        contour.save(os.path.join(output_dir, "contour", contour_name, f"contour_{name}.vtp"))

                    # Save polar projections
                    os.makedirs(os.path.join(output_dir, "planar_projections", name), exist_ok=True)
                    tracked_vessel.save_planar_projections(os.path.join(output_dir, "planar_projections", name))

                already_tracked_stopping.update_points(torch.tensor(results[name]["centerline"].points))
                pbar.update()

        return results
