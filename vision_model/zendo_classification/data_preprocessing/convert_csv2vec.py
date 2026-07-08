import csv
import torch
from pathlib import Path
from ast import literal_eval
from collections import defaultdict

color_lexicon = ["red", "blue", "yellow", "PAD"]
shape_lexicon = ["block", "wedge", "pyramid", "PAD"]
orientation_lexicon = ["upright", "upside_down", "flat", "cheesecake", "PAD"]
max_objects = 7
color_to_idx = {color: idx for idx, color in enumerate(color_lexicon)}
shape_to_idx = {shape: idx for idx, shape in enumerate(shape_lexicon)}
orientation_to_idx = {orientation: idx for idx, orientation in enumerate(orientation_lexicon)}
orientation_to_idx["doorstop"] = 2
token_PAD = len(color_lexicon) - 1
token_PAD_orientation = len(orientation_lexicon) - 1
token_PAD_rel = max_objects
token_NONE = max_objects + 1
# Root dataset folders
root = "../zendo-synthetic-data/test-dataset"
root_dirs = ["training23", "test1", "test2", "training25"]


directions = ["left", "right", "front", "back", "top", "bottom"]

for root_dir in root_dirs:
    root_path = Path(root) / Path(root_dir)
    csv_path = root_path / "ground_truth.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path}")
        continue

    print(f"Processing {csv_path}")
    scenes = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_name = Path(row["img_path"]).stem
            rule_id = scene_name.split("_")[0]

            object_id_str = row["object_name"]
            parts = object_id_str.split("_")
            _, shape, color = parts[:3]
            orientation = "_".join(parts[3:]).lower()

            try:
                touching_dict = literal_eval(row["touching"])
                touching_dict = {k: v for k, v in touching_dict.items() if v is not None}
            except Exception:
                touching_dict = {}

            pointing_target = row["pointing"]
            if pointing_target == "":
                pointing_target = None

            bbox = {
                "x": {
                    "min": int(row["image_x_min"]),
                    "max": int(row["image_x_max"]),
                },
                "y": {
                    "min": int(row["image_y_min"]),
                    "max": int(row["image_y_max"]),
                }
            }

            obj = {
                "ID": object_id_str,
                "color": color.lower(),
                "shape": shape.lower(),
                "orientation": orientation,
                "touching": touching_dict,
                "pointing": pointing_target,
                "bbox": bbox,
                "present": 1.0
            }

            scenes[(rule_id, scene_name)].append(obj)

    for (rule_id, scene_name), obj_list in scenes.items():

        id_map = {obj["ID"]: i for i, obj in enumerate(obj_list)}
        for obj in obj_list:
            obj["ID"] = id_map[obj["ID"]]

            obj["touching"] = {
                k: (id_map[v] if v in id_map else None)
                for k, v in obj.get("touching", {}).items()
            }

            pt = obj.get("pointing")
            obj["pointing"] = id_map[pt] if pt in id_map else None

        tensors = []
        for obj in obj_list:
            vec = []

            vec.append(obj["ID"])

            color = obj.get("color", "PAD")
            vec.append(color_to_idx.get(color, token_PAD))

            shape = obj.get("shape", "PAD")
            vec.append(shape_to_idx.get(shape, token_PAD))

            orientation = obj.get("orientation", "PAD")
            vec.append(orientation_to_idx.get(orientation, token_PAD_orientation))

            for dir in directions:
                target_id = obj.get("touching", {}).get(dir, token_NONE)
                vec.append(target_id)

            pointed_id = obj.get("pointing", token_NONE)
            if pointed_id is None:
                pointed_id = token_NONE
            vec.append(pointed_id)

            bb = obj.get("bbox", {})
            bb_features = [
                bb.get("x", {}).get("min", -1), 
                bb.get("x", {}).get("max", -1),
                bb.get("y", {}).get("min", -1), 
                bb.get("y", {}).get("max", -1)
            ]
            vec.extend(bb_features)

            tensors.append(torch.tensor(vec, dtype=torch.long))
        while len(tensors) < max_objects:
            pad_tensor = torch.tensor([
                token_PAD_rel, token_PAD, token_PAD, token_PAD_orientation,
                *[token_PAD_rel] * 6, token_PAD_rel, -1, -1, -1, -1
            ], dtype=torch.long)
            tensors.append(pad_tensor)

        encoded_scene = torch.stack(tensors)

        tensor_path = Path("../zendo-synthetic-data/test-dataset_pred") / Path(root_dir) / f"{scene_name}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded_scene, tensor_path)

    print(f"Done writing tensors for {root_dir}")
