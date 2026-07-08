import torch
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from zendo_detection.zendo_encoder import ZendoStructureEncoding
from evaluation.evaluate_tensors import evaluate_tensor_files

def evaluate_all_tensor_predictions(pred_root_dir, gt_root_dir, encoder, output_csv):
    pred_root_dir = Path(pred_root_dir)
    gt_root_dir = Path(gt_root_dir)
    subdirs = ["test", "training23", "training25"]
    all_rows = []

    for subdir in subdirs:
        print(f"\n--- Evaluating dataset: {subdir} ---")
        pred_dir = pred_root_dir / subdir
        gt_dir = gt_root_dir / subdir

        if not pred_dir.exists() or not gt_dir.exists():
            print(f"Skipping {subdir} - missing directories.")
            continue

        for pred_path in pred_dir.rglob("*.pt"):
            scene_name = pred_path.stem
            gt_path = gt_dir / f"{scene_name}.pt"

            if not gt_path.exists():
                print(f"Missing GT for {subdir}/{scene_name}")
                continue

            try:
                pred_tensor = torch.load(pred_path)
                gt_tensor = torch.load(gt_path)
                if pred_tensor.shape[0] != encoder.max_objects or gt_tensor.shape[0] != encoder.max_objects:
                    print(f"Invalid tensor shape for {subdir}/{scene_name} (Pred: {pred_tensor.shape}, GT: {gt_tensor.shape})")
                    continue
            except Exception as e:
                print(f"Failed to load tensors for {subdir}/{scene_name}: {e}")
                continue

            results, pred_len = evaluate_tensor_files(pred_tensor, gt_tensor, encoder, scene_name=scene_name)
            for i, result in enumerate(results):
                row = {
                    "dataset": subdir,
                    "scene": scene_name,
                    "object_id": i,
                    "predicted_length": pred_len,
                    "ground_truth_length": int((gt_tensor[:, 1] != encoder.token_PAD).sum().item()),
                    **result
                }
                all_rows.append(row)

    # Write to CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fieldnames = list(all_rows[0].keys()) if all_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    if not all_rows:
        print("No data processed.")
        return

    # Summary
    total = sum(1 for row in all_rows if row["object_id"] < encoder.max_objects)

    if total > 0:
        correct_counts = {k: sum(r[k] for r in all_rows) for k in all_rows[0] if k.endswith("_correct")}
        print("\n--- Accuracy Summary (Across All Datasets) ---")
        for k, v in correct_counts.items():
            print(f"{k}: {v}/{total} ({v / total:.2%})")

        # Length accuracy
        scene_roots = [row for row in all_rows if row["object_id"] == 0]
        correct_length_count = sum(1 for row in scene_roots if row["predicted_length"] == row["ground_truth_length"])
        total_scenes = len(scene_roots)
        print(f"length_correct: {correct_length_count}/{total_scenes} ({correct_length_count / total_scenes:.2%})")

        # Bar chart
        attr_keys = [
            "color_correct",
            "shape_correct",
            "orientation_correct",
            "touching_correct",
            "pointing_correct",
            "bbox_correct"
        ]

        correct_counts = {k: sum(r[k] for r in all_rows) for k in attr_keys}
        total = sum(1 for r in all_rows if r["object_id"] < encoder.max_objects)
        accuracy = {k: correct_counts[k] / total for k in attr_keys}
        attributes = list(accuracy.keys())
        values = list(accuracy.values())

        colors = ['#1cbd70', '#27cda8', '#539ee3', '#3d7be0', '#8253d8', '#b03cc2']

        plt.figure(figsize=(8, 4))
        plt.bar(attributes, values, color=colors[:len(attributes)])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Vision Model Accuracy by Attribute (All Datasets)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True, help="Root directory with GT subdirectories")
    parser.add_argument("--pred_dir", type=str, required=True, help="Root directory with predicted subdirectories")
    parser.add_argument("--output_csv", type=str, required=False, default="evaluation_fieldwise.csv", help="Where to save per-field comparison CSV")
    args = parser.parse_args()

    encoder = ZendoStructureEncoding(
        color_lexicon=["red", "blue", "yellow"],
        shape_lexicon=["block", "wedge", "pyramid"],
        orientation_lexicon=["upright", "upside_down", "flat", "cheesecake"],
        max_objects=7
    )

    evaluate_all_tensor_predictions(args.pred_dir, args.gt_dir, encoder, args.output_csv)
