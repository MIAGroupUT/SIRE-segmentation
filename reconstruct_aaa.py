import os

import nibabel as nib
import numpy as np
import pyvista as pv
import torch
import typer

from pqdm.processes import pqdm

from src.sire.inr import INR, INRSkeletonPointData
from src.sire.inr.losses import ManifoldLoss, NeuralPullLoss, SkeletonLoss
from src.sire.inr.models import Siren
from src.sire.reconstruct.vascular_model import VascularModel

app = typer.Typer()


def euler_characteristic(poly: pv.PolyData):
    return poly.n_points - poly.extract_all_edges().n_lines + poly.n_faces_strict


def cap_mesh(poly: pv.PolyData, cap_points: np.array):
    def generate_plane(points, anchor):
        centroid = np.mean(points, axis=0)
        radius = np.linalg.norm(points - centroid, axis=1).mean()

        centered_points = points - centroid
        _, _, Vt = np.linalg.svd(centered_points)
        normal = Vt[-1, :]

        if np.dot(normal, anchor - centroid) > 0:
            normal = -normal

        return pv.Plane(centroid, normal, i_size=2.5 * radius, j_size=2.5 * radius)

    poly_center = poly.points.mean(axis=0)
    poly = poly.clip_surface(generate_plane(cap_points, poly_center))
    poly = poly.fill_holes(50)

    return poly


def reconstruct(name: str, out_dir: str, contour: np.array, centerline: np.array, omega: int, cap: bool = True):
    losses = [(NeuralPullLoss(), 0.15), (ManifoldLoss(), 0.05), (SkeletonLoss(), 0.1)]

    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sdfs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)

    if len(contour.reshape(-1, 3)) < 500:
        print(f"{os.path.join(out_dir, name)}: Invalid contour")
        return 1

    inr_point_data = INRSkeletonPointData(
        torch.tensor(centerline), torch.tensor(contour.reshape(-1, 3)), norm_scale=1.6
    )
    inr_module = INR(Siren([3, 64, 64, 64, 64, 64, 64, 1], omega=omega), losses=losses, device="cpu")
    inr_module.fit(inr_point_data, n_points=1000, n_iters=5_000, verbose=False, plots=False)
    inr_module.export_to_onnx((1000, 3), os.path.join(out_dir, "models", f"{name}_{omega}.onnx"))

    poly, sdf = inr_point_data.reconstruct_pyvista(
        os.path.join(out_dir, "models", f"{name}_{omega}.onnx"), resolution=256, verbose=False
    )

    if cap:
        top_cap, bottom_cap = contour[0], contour[-1]
        poly = cap_mesh(poly, top_cap)
        poly = cap_mesh(poly, bottom_cap)

    if euler_characteristic(poly) == 2:
        poly.save(sample := os.path.join(out_dir, "meshes", f"{name}_{omega}.vtp"))
        nib.save(nib.Nifti1Image(sdf, affine=np.eye(4)), os.path.join(out_dir, "sdfs", f"{name}_{omega}.nii.gz"))
        print(f"Correct topology - saving sample: {sample}")

    return name, omega, euler_characteristic(poly)


@app.command()
def run_reconstruction(
    in_dir: str = typer.Option(
        "/Users/patrykrygiel/Documents/UTWENTE/Datasets/AAA/M3i/full-segmentation", "-i", "--in-dir"
    ),
    out_dir: str = typer.Option(
        "/Users/patrykrygiel/Documents/UTWENTE/Datasets/AAA/M3i/raw_reconstructions", "-o", "--out-dir"
    ),
    n_jobs: int = typer.Option(1, "-n", "--n-jobs"),
):
    # Define segments to be reconstructed
    segments = {
        "full": [12, 14, 16, 18, 20, 22, 24],
        "full-iliacs": [10, 12, 14, 16, 18, 20, 22, 24],
        "iliac_right": [2, 4, 6, 8],
        "iliac_left": [2, 4, 6, 8],
        "renal_left": [2, 3, 4, 5],
        "renal_right": [3, 4, 5],
    }

    # Define pruning
    pruning = {
        "renal_left": 4,
        "renal_right": 4,
        "iliac_left": 8,
        "iliac_right": 8,
    }

    args = []
    samples = [filename for filename in sorted(os.listdir(in_dir)) if filename != ".DS_Store"]

    for sample in samples:
        branches = [
            "_".join(filename.split("_")[1:]).split(".")[0]
            for filename in sorted(os.listdir(os.path.join(in_dir, sample, "contour", "lumen")))
            if filename != ".DS_Store"
        ]

        # Load vascular model from contour directory
        vascular_model = VascularModel.load_from_directory(
            os.path.join(in_dir, sample, "contour", "lumen"), filenames=branches
        )

        # Correct centerlines
        vascular_model.correct_overlap("abdominal_aorta", "renal_left", merge_tol=8)
        vascular_model.correct_overlap("abdominal_aorta", "renal_right", merge_tol=8)
        bifurcation_point = vascular_model.correct_bifurcation(
            "abdominal_aorta", "iliac_left", "iliac_right", merge_tol=3
        )

        # Get pruned contours
        contours = {
            name: vascular_model.get_pruned_contour(name, "abdominal_aorta", pruning[name])
            for name in branches
            if name != "abdominal_aorta"
        }
        contours["abdominal_aorta"] = (
            vascular_model.contours["abdominal_aorta"],
            vascular_model.contours["abdominal_aorta"].mean(axis=1),
        )

        # Save corrected centerlines
        for name in vascular_model.centerlines.keys():
            os.makedirs(os.path.join(out_dir, sample, "centerlines"), exist_ok=True)

            if "iliac" in name:
                centerline = vascular_model.get_corrected_contour(name, connect_to=bifurcation_point).mean(axis=1)
            else:
                centerline = vascular_model.get_corrected_contour(name).mean(axis=1)

            pv.PolyData(
                centerline, lines=np.array([[2, i, i + 1] for i in range(len(centerline) - 1)]).flatten().astype(int)
            ).save(os.path.join(out_dir, sample, "centerlines", f"{name}.vtp"))

        # Add segments for reconstruction
        for segment, omegas in segments.items():

            # Full configuration
            if segment == "full":
                args.extend(
                    [
                        (
                            "full",
                            os.path.join(out_dir, sample),
                            np.concatenate([contour[0] for name, contour in contours.items()]),
                            np.concatenate([contour[1] for name, contour in contours.items()]),
                            omega,
                            False,
                        )
                        for omega in omegas
                    ]
                )

            # Aorta + iliacs configuration
            elif segment == "full-iliacs":
                args.extend(
                    [
                        (
                            "full-iliacs",
                            os.path.join(out_dir, sample),
                            np.concatenate(
                                [contour[0] for name, contour in contours.items() if "aorta" in name or "iliac" in name]
                            ),
                            np.concatenate(
                                [contour[1] for name, contour in contours.items() if "aorta" in name or "iliac" in name]
                            ),
                            omega,
                            False,
                        )
                        for omega in omegas
                    ]
                )

            # Custom configuration through list of branches
            elif segment.startswith("list-"):
                args.extend(
                    [
                        (
                            segment,
                            os.path.join(out_dir, sample),
                            np.concatenate([contour[0] for name, contour in contours.items() if name in segment]),
                            np.concatenate([contour[1] for name, contour in contours.items() if name in segment]),
                            omega,
                            False,
                        )
                        for omega in omegas
                    ]
                )

            # Single segments
            elif segment in contours.keys():
                args.extend(
                    [
                        (
                            segment,
                            os.path.join(out_dir, sample),
                            contours[segment][0],
                            contours[segment][1],
                            omega,
                            True,
                        )
                        for omega in omegas
                    ]
                )

    pqdm(args, reconstruct, argument_type="args", n_jobs=n_jobs)


if __name__ == "__main__":
    app()
