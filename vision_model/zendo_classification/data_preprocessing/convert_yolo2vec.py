from pathlib import Path
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from zendo_detection.yolo_dataset import ZendoYOLODataset

def export_dataset_to_pt(dataset, output_dir):
    """
    Iterates over a ZendoYOLODataset and saves each sample's tensor to a .pt file.
    Each file is named after the image stem (without extension).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(dataset)):
        image_tensor, structure_tensor, img_path = dataset[i]
        stem = Path(img_path).stem
        tensor_path = output_dir / f"{stem}.pt"
        torch.save(structure_tensor, tensor_path)

    print(f"Exported {len(dataset)} tensors to {output_dir}")
dataset = ZendoYOLODataset("../../zendo_blocks_object_detection/zendo_yolo_dataset_including_labels")

export_dataset_to_pt(dataset, "encoded_scenes")