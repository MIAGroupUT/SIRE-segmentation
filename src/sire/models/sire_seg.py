import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from monai.losses import DiceLoss

from src.sire.data.polar_sampler import PolarSampler
from src.sire.losses.euclidean_loss import EuclideanLoss
from src.sire.losses.ht_losses import JSLoss
from src.sire.models.networks.dilation_net import DilationNet
from src.sire.models.sire_base import SIREBase
from src.sire.utils.dsnt import normalize_coords, spatial_expectation, spatial_softmax, unnormalize_coords


class SIRESegmentation(SIREBase):
    def __init__(
        self,
        out_channels: int = 1,
        n_rad: int = 128,
        n_angles: int = 64,
        dsnt_sigma: float = 0.1,
        lr: float = 0.001,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(lr, seed)

        self.n_rad = n_rad
        self.n_angles = n_angles
        self.sigma = torch.tensor([dsnt_sigma])

        self.polar_sampler = PolarSampler(n_rad, n_angles)
        self.head = DilationNet(channels=[1, 8, 16, 32, 64, 64, out_channels], dilations=[1, 2, 4, 8, 16, 32])

        self.euc_loss = EuclideanLoss()
        self.reg_loss = JSLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def get_scale(self, scales_ft: torch.tensor, scales: torch.tensor):
        scales_ft_max, _ = torch.max(scales_ft, dim=1)
        scales_weights = F.softmax(scales_ft_max, dim=1)
        return (scales_weights * scales[..., None]).sum(dim=1)

    def snap_coords(self, coords: torch.tensor):
        _, _, channels, _ = coords.shape

        for i in range(1, channels):
            coords[:, :, i] = coords[:, :, i - 1] + F.relu(coords[:, :, i] - coords[:, :, i - 1])

        return coords

    def extract_channels(self, data, coords_pred, coords_ht_pred, seg_pred):
        contour_types = [contour_type[0] for contour_type in data["global"]["contour_types"]]
        contour_id = (
            torch.tensor([contour_types.index(contour_type) for contour_type in data["sample"]["label"]])
            .to(coords_pred)
            .long()
        )

        coords_pred_i = torch.cat(
            [torch.index_select(unbatched, 1, index)[None] for unbatched, index in zip(coords_pred, contour_id)]
        )
        coords_ht_pred_i = torch.cat(
            [torch.index_select(unbatched, 1, index)[None] for unbatched, index in zip(coords_ht_pred, contour_id)]
        )
        seg_pred_i = torch.cat(
            [torch.index_select(unbatched, 0, index)[None] for unbatched, index in zip(seg_pred, contour_id)]
        )

        return coords_pred_i, coords_ht_pred_i, seg_pred_i

    def forward(self, data):
        # Extract polar image
        scales_ft = self.backbone(data)
        scales = self.get_scale(scales_ft, data["global"]["scales"])
        polar_images, polar_points, polar_mask = self.polar_sampler(data, scales, padding=self.head.padding)

        data["sample"]["polar_image"] = polar_images
        data["sample"]["polar_points"] = polar_points
        data["sample"]["polar_mask"] = polar_mask

        # Regress heatmaps and segmentation masks with the CNN network
        out = self.head(polar_images.unsqueeze(1))

        # Split into heatmaps and segmentation
        all_channels = out.shape[1]
        ht, seg = out[:, : all_channels // 2], out[:, all_channels // 2 :]

        # Process heatmaps
        ht = seg.transpose(1, 2).contiguous()
        batch_size, n_angles, channels, n_rad = ht.shape

        # Extract coordinates from the heatmap
        ht_soft = spatial_softmax(ht.view(-1, channels, n_rad))
        coords = spatial_expectation(ht_soft)

        # Reshape back to batch
        ht_soft = ht_soft.view(batch_size, n_angles, channels, n_rad)
        coords = coords.view(batch_size, n_angles, channels, 1)

        # Unnormalize the predicted coords and snap to not intersect
        coords = unnormalize_coords(coords, self.n_rad) - self.n_rad / 2
        # coords = self.snap_coords(coords)

        return ht_soft, coords, seg, scales

    def get_losses(self, data, coords_pred, coords_ht_pred, seg_pred, scales_pred):
        center, normal = data["sample"]["center"], data["sample"]["normal"]
        coords_true, seg_true = data["sample"]["polar_points"].float(), data["sample"]["polar_mask"].float()
        _, _, num_channels, dim = coords_true.shape

        # Extract channel based on present label for the loss calculation
        coords_pred, coords_ht_pred, seg_pred = self.extract_channels(data, coords_pred, coords_ht_pred, seg_pred)

        # Dice loss (on lower half only)
        dice_loss = self.dice_loss(seg_pred[..., self.n_rad // 2 :], seg_true[..., self.n_rad // 2 :].unsqueeze(1))

        # Euclidean loss
        padding = self.head.padding
        coords_pred_cart = self.polar_sampler.inverse(coords_pred, center.float(), normal.float(), scales_pred, padding)
        coords_true_cart = self.polar_sampler.inverse(coords_true, center.float(), normal.float(), scales_pred, padding)
        euc_loss = self.euc_loss(coords_pred_cart.float(), coords_true_cart.float())

        # Regularization on heatmaps (for each radius seperately)
        coords_pred = normalize_coords(coords_pred + self.n_rad / 2, self.n_rad)
        coords_true = normalize_coords(coords_true + self.n_rad / 2, self.n_rad)
        reg_loss = self.reg_loss(
            coords_ht_pred.view(-1, num_channels, self.n_rad),
            coords_true.view(-1, num_channels, dim).float(),
            self.sigma.to(coords_ht_pred),
        )

        return euc_loss, reg_loss, dice_loss

    def shared_step(self, data, batch_idx, stage):
        coords_ht_pred, coords_pred, seg_pred, scales_pred = self(data)

        euc_loss, reg_loss, dice_loss = self.get_losses(data, coords_pred, coords_ht_pred, seg_pred, scales_pred)

        # Total loss
        loss = euc_loss + reg_loss + dice_loss

        # Logger
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}/euc_loss", euc_loss, on_epoch=True, logger=True)
        self.log(f"{stage}/reg_loss", reg_loss, on_epoch=True, logger=True)
        self.log(f"{stage}/dice_loss", dice_loss, on_epoch=True, logger=True)

        coords_true = data["sample"]["polar_points"].float()
        self._plot_wandb_sample(coords_pred, scales_pred, coords_true, data, batch_idx, stage)

        return loss

    def _get_diameter(self, data, channel: int = 0, index: int = 0):
        center = data["sample"]["center"][index]
        points = data["sample"]["contours"][index, :, channel]

        return 2 * torch.linalg.norm(points - center.unsqueeze(0), dim=1).mean()

    def _plot_wandb_sample(self, coords_pred, scale_pred, coords_true, data, batch_idx, stage, index: int = 0):
        self.log(f"{stage}/scale", scale_pred[index], on_epoch=True, logger=True)
        self.log(f"{stage}/diameter", self._get_diameter(data, index=index), on_epoch=True, logger=True)
        self.log(
            f"{stage}/scale_diameter_ratio",
            scale_pred[index] / (self._get_diameter(data, index=index) + 1e-12),
            on_epoch=True,
            logger=True,
        )

        if stage == "val" and batch_idx == 0:
            label = data["sample"]["label"][index]

            n_rad = self.n_rad
            padding = self.head.padding

            coords_true = coords_true[index].cpu().numpy() + n_rad / 2 + padding
            coords_pred = coords_pred[index].cpu().numpy() + n_rad / 2 + padding

            fig, ax = plt.subplots()
            image = data["sample"]["polar_image"][index].T.cpu().numpy()
            W, H = image.shape

            ax.imshow(image, cmap="gray")
            ax.plot(
                torch.arange(len(coords_true)) + padding,
                coords_true.squeeze(),
                color="tab:green",
                label=f"GT ({label})",
            )

            colors = ["tab:red", "tab:orange"]
            contour_types = [contour_type[0] for contour_type in data["global"]["contour_types"]]

            for i, contour_type in enumerate(contour_types):
                ax.plot(
                    torch.arange(len(coords_pred)) + padding,
                    coords_pred[:, i].squeeze(),
                    color=colors[i],
                    label=contour_type,
                )
                ax.axis("off")

            rect = patches.Rectangle(
                (padding, padding),
                W - 2 * padding,
                H - 2 * padding,
                linewidth=1,
                edgecolor="tab:blue",
                facecolor="none",
            )
            ax.add_patch(rect)

            plt.legend()

            self.logger.log_image(key=f"media/val/{label}", images=[fig])
            plt.close()
