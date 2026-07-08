import copy
from pathlib import Path
from train_model import run_training
import optuna

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
         "../dataset/training15", "../dataset/training16", "../dataset/training17"
    ],
    "val_percent": 0.1,
    "image_size": [480, 640],
    "batch_size": 4,

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
    "num_epochs": 100,
    "dropout": 0.23,

    "weight_path": "zendo_model.pt",
    "scheduler_path": "scheduler.pt",

    # Head weights
    "classification": 0.126,
    "relations": 0.12,
    "bbox": 0.00997,

    #Architecture
    "layers": 4,
    "color_mult_layer": False,
    "shape_mult_layer": False,
    "orientation_mult_layer": False,
    "presence_mult_layer": False,
    "pointing_mult_layer": False,
    "touching_mult_layer": False,
    "bbox_mult_layer": False,

}

def objective(trial):
    cfg = copy.deepcopy(base_cfg)
    cfg["bbox_mult_layer"] = trial.suggest_categorical("bbox_mult_layer", [True, False])
    cfg["pointing_mult_layer"] = trial.suggest_categorical("pointing_mult_layer", [True, False])
    cfg["touching_mult_layer"] = trial.suggest_categorical("touching_mult_layer", [True, False])

    output_dir = Path("optuna_experiments_3") / f"trial_{trial.number:03d}"
    cfg["path"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_loss = run_training(cfg)  # This should return final validation loss
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=6)

print("Best hyperparameters:", study.best_params)
