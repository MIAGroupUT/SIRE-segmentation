from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.sire.inference.utils.step_schedulers import ConstantStepScheduler, StepSchedulerBase
from src.sire.inference.utils.stopping_criterions import StoppingCriterionBase


@dataclass
class VesselConfig:
    name: str
    seed_point: np.array
    segment_every_n_steps: int = 1
    step_scheduler: StepSchedulerBase = ConstantStepScheduler(1)
    stopping_criterions: Tuple[StoppingCriterionBase] = ()
