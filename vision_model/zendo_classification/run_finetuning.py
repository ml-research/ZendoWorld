import random
import copy
from pathlib import Path
from finetune_model import run_training

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
    "dataset_root": "../../zendo_blocks_object_detection/zendo_yolo_dataset_including_labels",
    "pretrained_weights": "zendo_model.pt",
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
cfg["path"] = str(base_output / f"finetune_{seed}")
print(f"Running experiment with seed {seed}, output path: {cfg['path']}")
run_training(cfg)
