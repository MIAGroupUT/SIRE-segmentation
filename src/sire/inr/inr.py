from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
from tqdm.auto import tqdm

from src.sire.inr.data import INRPointData


class INR:
    def __init__(self, model: nn.Module, losses: List[Any], device: str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.losses = [loss for loss, _ in losses]
        self.betas = [beta for _, beta in losses]

        self.betas = torch.tensor(self.betas).to(self.device)

    def _step(self, data: INRPointData, n_points: int, noise_var: float):
        sampled_data = data(n_points, noise_var)

        out_data = {}

        for query_name, query_dict in sampled_data.items():
            sdf, coords = self.model(query_dict["coords"])
            query_dict["sdf"] = sdf
            query_dict["coords"] = coords

            out_data[query_name] = query_dict

        loss_dict = {loss.__class__.__name__: loss(data.points, out_data) for loss in self.losses}
        total_loss = (self.betas * torch.stack(list(loss_dict.values()))).sum()

        return total_loss, loss_dict

    def fit(
        self,
        data: INRPointData,
        n_iters: int = 10000,
        n_points: int = 1000,
        noise_var: float = 0.1,
        lr: float = 3e-4,
        verbose: bool = True,
        plots: bool = False,
    ):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, n_iters // 3, 0.5)

        loss_vals = []

        with tqdm(total=n_iters, desc="Fitting INR", leave=False, disable=not verbose) as pbar:
            for _ in range(n_iters):
                optimizer.zero_grad()

                total_loss, loss_dict = self._step(data, n_points, noise_var)

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update()
                pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})

                list(loss_dict.keys())
                loss_vals.append(torch.stack(list(loss_dict.values())))

        if plots:
            loss_vals = torch.stack(loss_vals)
            loss_names = loss_dict.keys()

            for loss_name, loss_val in zip(loss_names, loss_vals.T):
                plt.plot(np.arange(len(loss_val)), loss_val.detach().cpu().numpy())
                plt.title(loss_name)
                plt.grid(True)
                plt.yscale("log")
                plt.show()

        return loss_dict

    def save_weights(self, output_path: str):
        torch.save(self.model.state_dict(), output_path)

    def export_to_onnx(self, input_shape: Tuple[int], output_path: str, dtype: torch.dtype = torch.float32):
        input_tensor = torch.rand(input_shape, dtype=dtype).to(self.device)
        torch.onnx.export(
            self.model,
            input_tensor,
            output_path,
            input_names=["input_coords"],
            output_names=["output_sdf", "output_coords"],
            dynamic_axes={"input_coords": {0: "num_points"}, "output_coords": {0: "num_points"}},
        )

    @classmethod
    def load_pretrained(cls, model_path: str, model: nn.Module, losses: List[Any], device: str = "cuda"):
        model.load_state_dict(torch.load(model_path))
        return cls(model, losses, device)
