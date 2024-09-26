from typing import List

import torch.nn as nn


class DilationNet(nn.Module):
    def __init__(self, channels: List[int], dilations: List[int], kernel_size: int = 3):
        super().__init__()

        assert len(channels) - 1 == len(dilations)
        self.padding = sum(dilations)

        self.model = nn.Sequential(
            *[
                self._build_block(channels_in, channels_out, kernel_size, dilation)
                for channels_in, channels_out, dilation in zip(channels[:-2], channels[1:-1], dilations[:-1])
            ],
            nn.Conv2d(channels[-2], channels[-1], kernel_size, dilation=dilations[-1]),
        )

    def _build_block(self, channels_in: int, channels_out: int, kernel_size: int, dilation: int):
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, dilation=dilation),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)
