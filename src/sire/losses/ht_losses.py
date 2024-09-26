import torch
import torch.nn as nn

from src.sire.utils.dsnt import render_gaussian


class JSLoss(nn.Module):
    def __init__(self):
        super(JSLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def _js(self, p: torch.tensor, q: torch.tensor, eps: float = 1e-24):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q) + eps).log()
        return 0.5 * (self.kl(m, (p + eps).log()) + self.kl(m, (q + eps).log()))

    def __call__(self, ht: torch.tensor, mean: torch.tensor, sigma: torch.tensor):
        size = ht.shape[2:]
        gauss = render_gaussian(mean, sigma, size)

        return self._js(ht, gauss)
