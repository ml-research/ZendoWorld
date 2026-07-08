from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv
from ast import literal_eval
from pathlib import Path

class ZendoImageToStructureDataset(Dataset):
    def __init__(self, encoder, root_paths, transform=None):
        self.samples = []
        self.encoder = encoder
        scenes = {}
        for root_path in root_paths:
            csv_path = Path(root_path) / "ground_truth.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"No ground_truth.csv found in {root_path}")
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    scene_name =  str(Path(root_path) / Path(*Path(row["img_path"]).parts[1:]))
                    if scene_name not in scenes:
                        scenes[scene_name] = {
                            "objects": [],
                            "image_path": str(Path(root_path) / Path(*Path(row["img_path"]).parts[1:]))
                        }

                    object_id = row["object_name"]
                    parts = object_id.split('_')
                    _, shape, color = parts[:3]
                    orientation = '_'.join(parts[3:])

                    try:
                        touching_dict = literal_eval(row["touching"])
                        touching_dict = {k: v for k, v in touching_dict.items() if v is not None}
                    except Exception:
                        touching_dict = {}

                    pointing_target = row["pointing"].strip()

                    bbox = {
                        "x": {
                            "min": int(row["image_x_min"]),
                            "max": int(row["image_x_max"]),
                        },
                        "y": {
                            "min": int(row["image_y_min"]),
                            "max": int(row["image_y_max"]),
                        },
                    }

                    scenes[scene_name]["objects"].append({
                        "ID": object_id,
                        "color": color.lower(),
                        "shape": shape.lower(),
                        "orientation": orientation.lower(),
                        "touching": touching_dict,
                        "pointing": pointing_target,
                        "bbox": bbox,
                    })

        self.scenes = list(scenes.values())
        print(f"Loaded {len(self.scenes)} scenes from {len(root_paths)} root paths.")
        self.transform = transform

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        img_path = scene["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        structure_tensor = self.encoder.encode_structure(scene)
        return image_tensor, structure_tensor, img_path