import csv
from pathlib import Path
from collections import defaultdict

def get_best_loss_from_csv(csv_path):
    best_loss = float("inf")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                loss = float(row["loss"])
                if loss < best_loss:
                    best_loss = loss
            except (ValueError, KeyError):
                continue
    return best_loss

def analyze_experiments(base_dir="experiments", top_k=5):
    base_path = Path(base_dir)
    assert base_path.exists(), f"Path not found: {base_path.resolve()}"

    results_by_seed = defaultdict(list)

    for seed_dir in base_path.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("run_seed_"):
            seed = seed_dir.name.split("_")[-1]
            for csv_dir in seed_dir.rglob("val_log.csv"):
                loss = get_best_loss_from_csv(csv_dir)
                if loss < float("inf"):
                    results_by_seed[seed].append((loss, csv_dir.parent))

    # Sort and show top configs per seed
    for seed, results in results_by_seed.items():
        print(f"Top {top_k} configs for seed {seed}")
        results.sort()  # sort by loss
        for rank, (loss, path) in enumerate(results[:top_k], 1):
            print(f"{rank}. Loss: {loss:.4f} | Config: {path.relative_to(base_path)}")

if __name__ == "__main__":
    analyze_experiments()
