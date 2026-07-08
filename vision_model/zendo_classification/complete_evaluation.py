from email.headerregistry import DateHeader
import torch
import argparse
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from zendo_detection.yolo_dataset import ZendoYOLODataset
from zendo_detection.zendo_encoder import ZendoStructureEncoding
from zendo_detection.model import ZendoImageToVectorModel
from evaluation.evaluate_tensors import evaluate_tensor_files
from zendo_detection.hungarian_loss import permutation_invariant_object_loss
from zendo_detection.zendo_encoder import ZendoStructureEncoding
from zendo_detection.model import ZendoImageToVectorModel, ZendoLightweightModel, ZendoSimpleModel
from zendo_detection.train import train_model
from zendo_detection.zendo_dataset import ZendoImageToStructureDataset

max_objects = 7
threshold = 0.5  # presence threshold

# Lexicons
color_lexicon = ["red", "blue", "yellow"]
shape_lexicon = ["block", "wedge", "pyramid"]
orientation_lexicon = ["upright", "upside_down", "flat", "cheesecake"]

# Token indices
token_PAD = len(color_lexicon)       # e.g., 3
token_PAD_shape = len(shape_lexicon) # e.g., 3
token_PAD_orientation = len(orientation_lexicon) # e.g., 4
token_PAD_rel = 7  # assuming 0-6 are object IDs, 7 is self, 8 is NONE
token_NONE = 8

# Config
cfg = {
    "path": "",
    "seed": 42,

    # Model
    "max_objects": 7,
    "token_dim": 384,

    # Lexicons
    "color_lexicon": ["red", "blue", "yellow"],
    "shape_lexicon": ["block", "wedge", "pyramid"],
    "orientation_lexicon": ["upright", "upside_down", "flat", "cheesecake"],

    # Data
    "dataset_root": "../../zendo_blocks_object_detection/zendo_yolo_dataset_including_labels",
    "val_percent": 0.1,
    "image_size": [480, 640],
    "batch_size": 16,

    # Augmentations
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.05
    },

    # Optimization
    "lr": 5.8e-5,
    "weight_decay": 7e-3,
    "lr_step_size": 10,
    "lr_gamma": 0.65,
    "num_epochs": 75,
    "dropout": 0.23,

    "weight_path": "zendo_model.pt",
    "scheduler_path": "scheduler.pt",

    # Head weights
    "classification": 0.126,
    "relations": 0.12,
    "bbox": 0.00997,

    "layers": 4,
    "color_mult_layer": False,
    "shape_mult_layer": False,
    "orientation_mult_layer": False,
    "presence_mult_layer": False,
    "pointing_mult_layer": True,
    "touching_mult_layer": True,
    "bbox_mult_layer": True,
}

def evaluate_model(dataset_paths, model_path, batch_size=16, output_csv="eval_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ZendoYOLODataset(cfg["dataset_root"])
    print(f"Total number of evaluation images: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = ZendoImageToVectorModel(
        cfg,
        num_output_tokens=cfg["max_objects"],
        token_dim=cfg["token_dim"],
        max_objects=cfg["max_objects"],
        num_colors=len(cfg["color_lexicon"]) + 1,
        num_shapes=len(cfg["shape_lexicon"]) + 1,
        num_orientations=len(cfg["orientation_lexicon"]) + 1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.to(device)
    model.eval()

    # Aggregation
    loss_totals = {
        "total": 0.0, "color": 0.0, "shape": 0.0, "orient": 0.0,
        "point": 0.0, "touch": 0.0, "bbox": 0.0, "presence": 0.0
    }
    loss_count = 0

    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            losses = permutation_invariant_object_loss(
                outputs=outputs,
                paths=paths,
                targets=targets,
                device=device,
                num_classes_dict={
                    "color": outputs["color"].shape[-1],
                    "shape": outputs["shape"].shape[-1],
                    "orientation": outputs["orientation"].shape[-1],
                    "pointing": outputs["pointing"].shape[-1],
                },
                head_weights={
                    "classification": cfg["classification"],
                    "relations": cfg["relations"],
                    "bbox": cfg["bbox"],
                }
            )

            for k, v in zip(loss_totals.keys(), losses):
                loss_totals[k] += v.item()
            loss_count += 1

    # Report
    if loss_count > 0:
        print("\n--- Loss Summary ---")
        for k, v in loss_totals.items():
            print(f"{k}_loss: {v / loss_count:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--gt_dir", type=str, help="Path to directory of ground truth tensors")
    parser.add_argument("--model", type=str, default="zendo_model.pt", help="Path to trained model")
    args = parser.parse_args()

    evaluate_model(["../test-dataset/test", "../test-dataset/training23", "../test-dataset/training25"], args.model)
