import glob
import os

import typer

from tqdm.auto import tqdm

from aaa_wrapper import AAATotalSegmentatorProcessor
from src.sire.inference.inference_models import SegmentationInferenceModel, TrackerInferenceModel
from src.sire.inference.segmentator_tracker import SegmentatorTrackerPipeline
from src.sire.models.sire_seg import SIRESegmentation
from src.sire.models.sire_tracker import SIRETracker

app = typer.Typer()


@app.command()
def inference(
    root_dir: str = typer.Option("/Users/patrykrygiel/Documents/UTWENTE/Datasets/AAA/UT/mhd", "-r", "--root-dir"),
    output_dir: str = typer.Option("results/test", "-o", "--output-dir"),
    device: str = typer.Option("cpu", "-d", "--device"),
):
    samples = os.listdir(root_dir)

    # Load tracking model
    tracker_model = TrackerInferenceModel(
        model=SIRETracker.load_from_checkpoint("src/sire/models/checkpoints/tracking_model.ckpt"),
        scales=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        npoints=32,
        subdivisions=3,
        device=device,
    )

    # Load segmentation model
    segmentation_models = [
        SegmentationInferenceModel(
            model=SIRESegmentation.load_from_checkpoint("src/sire/models/checkpoints/segmentation_model.ckpt"),
            names=["lumen"],
            scales=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
            npoints=32,
            subdivisions=2,
            device=device,
        ),
    ]

    # Prepare pipeline
    aaa_preprocessor = AAATotalSegmentatorProcessor()
    tracker_pipeline = SegmentatorTrackerPipeline(tracker_model, segmentation_models)

    # Running for each sample
    for sample in tqdm(samples, desc="Samples"):
        os.makedirs(os.path.join(output_dir, sample), exist_ok=True)
        image_path = glob.glob(os.path.join(root_dir, f"{sample}/*.mhd"))[0]
        sample_dir = os.path.join(output_dir, sample)

        vessel_configs = aaa_preprocessor(image_path, sample_dir, device=device if device != "cuda" else "gpu")
        tracker_pipeline.run(
            image_path, output_dir=sample_dir, vessel_configs=vessel_configs, already_tracked_distance=0
        )


if __name__ == "__main__":
    app()
