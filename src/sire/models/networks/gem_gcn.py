import torch
import torch.nn as nn
import torch.nn.functional as F

from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from torch.nn import Sequential


class GEMGCN(torch.nn.Module):
    def __init__(self, channels=16, out_channels=1, convs=3, n_rings=2, max_order=2):
        """
        OnionNet implementation of the GEMGCN used in the Faust experiment in the ICLR paper.
        nverts: number of vertices in the onion
        """
        super(GEMGCN, self).__init__()

        # onion structure
        self.out_channels = out_channels

        kwargs = dict(
            n_rings=n_rings, band_limit=max_order, num_samples=7, checkpoint=False, batch=100000, batch_norm=False
        )

        model = [GemResNetBlock(32, channels, 0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [GemResNetBlock(channels, channels, max_order, max_order, **kwargs)]
        model += [GemResNetBlock(channels, channels, max_order, 0, **kwargs)]

        self.model = Sequential(*model)

        # Dense final layer
        self.lin1 = nn.Linear(channels, out_channels)

    def forward(self, data, nverts: int = 642, num_scales: int = None):
        attr0 = (data.edge_index, data.precomp, data.connection)

        x = data.features.reshape(-1, 32, 1).float()

        for layer in self.model:
            x = layer(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]
        x = F.relu(self.lin1(x)).squeeze()

        if num_scales is not None:
            all_nverts = nverts * num_scales * self.out_channels

            x = torch.cat(
                [
                    x[i * all_nverts : (i + 1) * all_nverts].view(num_scales, -1, self.out_channels)
                    for i in range(len(x) // all_nverts)
                ],
                dim=1,
            ).transpose(0, 1)
        return x
