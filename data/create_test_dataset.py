import re
import pickle
import torch
from pathlib import Path
from collections import defaultdict

# -----------------------
# Utils
# -----------------------

def split_csv_line(line):
    pattern = r'[^,"]+|"(?:\\.|[^"\\])*"'
    return re.findall(pattern, line)

def remove_generate_valid_structure(query):
    match = re.match(
        r'\"generate_(valid|invalid)_structure\(\s*\[(.*)\]\s*,\s*Structure\s*\)\"',
        query.strip()
    )
    if match:
        return match.group(2).strip()
    else:
        return query

def load_tensor(path):
    return torch.load(path, weights_only=True)

# -----------------------
# Config
# -----------------------

data_dirs = ["training23", "test1", "test2", "training25"]

data_root = Path("../zendo-synthetic-data")
csv_root = Path("test-dataset")
tensor_root = Path("test-dataset_pred")

# -----------------------
# Step 1: Group rows by scene
# -----------------------

scene_groups = defaultdict(list)

for data_dir in data_dirs:
    csv_path = data_root / csv_root / data_dir / "ground_truth.csv"
    print(f"Processing {csv_path}...")

    with open(csv_path, "r") as f:
        header = f.readline()

        for line in f:
            parts = split_csv_line(line)

            scene_name = parts[0]
            rule_idx = scene_name.split('_')[0]
            rule_key = (data_dir, rule_idx)

            is_negative = scene_name.endswith("_n")
            label = 0 if is_negative else 1

            tensor_path = data_root / tensor_root / data_dir / (scene_name + ".pt")
            image_path = data_root / csv_root / data_dir / rule_idx / (scene_name + ".png")

            if not tensor_path.exists():
                continue

            try:
                program_query = remove_generate_valid_structure(parts[3])

                scene_groups[(rule_key, scene_name)].append({
                    "tensor_path": tensor_path,
                    "label": label,
                    "program": program_query,
                    "image_path": image_path
                })

            except Exception as e:
                print(f"Error parsing row: {e}")

print(f"Grouped into {len(scene_groups)} scenes")

# -----------------------
# Step 2: Collapse scenes → examples
# -----------------------

rule_to_examples = defaultdict(list)

for (rule_key, scene_name), rows in scene_groups.items():
    row = rows[0]  # all rows share same tensor/image

    try:
        tensor = load_tensor(row["tensor_path"])

        rule_to_examples[rule_key].append((
            tensor,
            row["label"],
            row["program"],
            row["image_path"]
        ))

    except Exception as e:
        print(f"Error loading tensor for {scene_name}: {e}")

print(f"Collected examples for {len(rule_to_examples)} rules")

# -----------------------
# Step 3: Build tasks
# -----------------------

tasks = []

for (data_dir, rule_idx), examples in rule_to_examples.items():

    # --- Deduplicate by image path ---
    seen = set()
    deduped = []
    for ex in examples:
        key = str(ex[3])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    if len(deduped) < 20:
        continue

    # --- Split labels ---
    positives = [ex for ex in deduped if ex[1] == 1]
    negatives = [ex for ex in deduped if ex[1] == 0]

    if len(positives) < 10 or len(negatives) < 10:
        print(f"Skipping rule {rule_idx}: not enough pos/neg")
        continue

    # --- Select examples ---
    chosen = positives[:10] + negatives[:10]

    rule_query = chosen[0][2]

    tasks.append([
        rule_query,
        [(tensor, label) for (tensor, label, *_rest) in chosen],
        [image_path for (*_rest, image_path) in chosen],
    ])

print(f"Prepared {len(tasks)} tasks")

# -----------------------
# Save dataset
# -----------------------

output_path = "data/full_test_dataset.pkl"

with open(output_path, "wb") as f:
    pickle.dump(tasks, f)

print(f"Saved dataset to {output_path}")