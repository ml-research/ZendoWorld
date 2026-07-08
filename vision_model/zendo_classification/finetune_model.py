import torch
import yaml
import time
import random
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from zendo_detection.zendo_encoder import ZendoStructureEncoding
from zendo_detection.model import ZendoImageToVectorModel
from zendo_detection.train import train_model
from zendo_detection.yolo_dataset import ZendoYOLODataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(cfg: dict):
    torch.cuda.empty_cache()
    set_seed(cfg["seed"])
    output_path = Path(cfg["path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "meta.yaml", "w") as f:
        yaml.dump(cfg, f)

    dataset = ZendoYOLODataset(cfg["dataset_root"])
    print(dataset.__getitem__(0)[1:])
    num_val = int(len(dataset) * cfg["val_percent"])
    num_train = len(dataset) - num_val
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

    train_transform = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
    ])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    print(f"Training on {len(train_loader.dataset)} samples, validation on {len(val_loader.dataset)} samples")

    model = ZendoImageToVectorModel(
        config=cfg,
        num_output_tokens=cfg["max_objects"],
        token_dim=cfg["token_dim"],
        dropout=cfg["dropout"],
    )

    weight_path = Path(cfg["pretrained_weights"])
    if weight_path.exists():
        print(f"Loading pretrained weights from {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("Loaded pretrained weights (fine-tuning).")
    else:
        print("No pretrained weights found. Starting from scratch.")

    num_training_steps = cfg["num_epochs"] * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if Path(cfg["scheduler_path"]).exists():
        scheduler.load_state_dict(torch.load(cfg["scheduler_path"], map_location="cpu"))
        print(f"Loaded scheduler from {cfg['scheduler_path']}")
    else:
        print("No scheduler found. Starting from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Number of training examples: {len(dataset)}")

    start_wall_time = time.time()
    start_cpu_time = time.process_time()

    head_weights = {
        "classification": cfg["classification"],
        "relations": cfg["relations"],
        "bbox": cfg["bbox"],
    }

    best_weights, best_loss = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        cfg["path"],
        num_epochs=cfg["num_epochs"],
        head_weights=head_weights,
        scheduler=scheduler,
    )

    end_wall_time = time.time()
    end_cpu_time = time.process_time()

    if best_weights is not None:
        torch.save(best_weights, output_path / "zendo_model.pt")
        print("Saved best model (early stopping or final epoch).")
    else:
        torch.save(model.state_dict(), output_path / "zendo_model.pt")
        print("Model not improved? Saved last state.")

    torch.save(scheduler.state_dict(), output_path / "scheduler.pt")

    print(f"Model saved to {output_path}")
    print(f"Total wall time: {end_wall_time - start_wall_time:.2f} seconds")
    print(f"Total CPU time: {end_cpu_time - start_cpu_time:.2f} seconds")
    return best_loss
