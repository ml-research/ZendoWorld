import random
import copy
from pathlib import Path
from train_model import run_training

# Base configuration
base_cfg = {
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
    "dataset_paths": [
        "../../zendo-synthetic-data/dataset/training2","../../zendo-synthetic-data/dataset/training6", "../../zendo-synthetic-data/dataset/training7", "../../zendo-synthetic-data/dataset/training8",
        "../../zendo-synthetic-data/dataset/training9", "../../zendo-synthetic-data/dataset/training10", "../../zendo-synthetic-data/dataset/training11", "../../zendo-synthetic-data/dataset/training12", 
        "../../zendo-synthetic-data/dataset/training13", "../../zendo-synthetic-data/dataset/training14", 
        "../../zendo-synthetic-data/dataset/training15", "../../zendo-synthetic-data/dataset/training16",
        "../../zendo-synthetic-data/dataset/training17",
        "../../zendo-synthetic-data/dataset/training18",
        "../../zendo-synthetic-data/rules-dataset/training19",
        "../../zendo-synthetic-data/rules-dataset/training20",
        "../../zendo-synthetic-data/rules-dataset/training21",
        "../../zendo-synthetic-data/rules-dataset/training22",
        "../../zendo-synthetic-data/rules-dataset/training24",
        "../../zendo-synthetic-data/rules-dataset/training26"
    ],
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

# Run 3 experiments with random seeds
base_output = Path("experiments")
base_output.mkdir(exist_ok=True)
seed = random.randint(0, 10000)
cfg = copy.deepcopy(base_cfg)
cfg["seed"] = seed
cfg["path"] = str(base_output / f"experiment_{seed}")
run_training(cfg)
