class StepSchedulerBase:
    def __init__(self, step_size: float):
        self.step_size = step_size

    def __call__(self, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class ConstantStepScheduler(StepSchedulerBase):
    def __call__(self, **kwargs):
        return self.step_size

    def __str__(self):
        return f"{self.__class__.__name__}(step_size={self.step_size})"


class AdaptiveStepScheduler(StepSchedulerBase):
    def __init__(self, step_size: float, scale_ratio: float = 0.2):
        super().__init__(step_size)
        self.scale_ratio = scale_ratio

    def __call__(self, scale: int, **kwargs):
        if scale is not None:
            self.step_size = self.scale_ratio * (scale.item() / 10)

        return self.step_size

    def __str__(self):
        return f"{self.__class__.__name__}(step_size={self.step_size}, scale_ratio={self.scale_ratio})"
