import os
import yaml
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ZendoYOLODataset(Dataset):
    def __init__(self, root_dir, transform=None, max_objects=7):
        self.root_dir = Path(root_dir)
        yaml_path = self.root_dir / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"No data.yaml found in {self.root_dir}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.names = data["names"]

        self.token_PAD = 3
        self.token_PAD_orientation = 4
        self.token_PAD_rel = max_objects
        self.token_NONE = max_objects + 1
        self.max_objects = max_objects

        self.samples = []
        for split in ["test"]:
            img_dir = self.root_dir / split / "images_cropped_asym"
            label_dir = self.root_dir / split / "labels"
            if not img_dir.exists() or not label_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                label_path = label_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    continue

                # --- do parsing immediately ---
                raw_pieces = self._load_labels(label_path)
                objects = self._build_objects(raw_pieces, self.names)
                if len(objects) > max_objects:
                    print(f"Skipping {img_path} with {len(objects)} objects (>{max_objects})")
                    continue
                if len(objects) == 0:
                    print(f"Skipping {img_path} with {len(objects)} objects")
                    continue

                structure_tensor = self._scene_to_tensor(objects, img_path)
                self.samples.append((img_path, structure_tensor))

        print(f"Loaded {len(self.samples)} valid samples from {self.root_dir}")
        self.transform = transform or transforms.ToTensor()

    def _build_objects(self, raw_pieces, names):
        objects, next_id = [], 0
        for cls_idx, x, y, w, h in raw_pieces:
            name = names[cls_idx]
            parsed_pieces = self._parse_name(name)

            # Convert YOLO → bbox
            x_min, y_min = x - w / 2, y - h / 2
            x_max, y_max = x + w / 2, y + h / 2

            prev_id = None
            for attrs in parsed_pieces:
                color_id, shape_id, orientation_id = self._map_to_ints(attrs)
                obj = {
                    "ID": next_id,
                    "color": attrs["color"],
                    "shape": attrs["shape"],
                    "grounded": attrs["grounded"],
                    "orientation": attrs["orientation"],
                    "color_id": color_id,
                    "shape_id": shape_id,
                    "orientation_id": orientation_id,
                    "bbox": (x_min, y_min, x_max, y_max),
                }
                if prev_id is not None:
                    obj["stacked_on"] = prev_id
                objects.append(obj)
                prev_id = next_id
                next_id += 1
        return objects

    def __len__(self):
        return len(self.samples)

    def _parse_name(self, name):
        """
        Parse a class name into one or two pieces.
        Example:
          'Blue_Block_Grounded_Upright'
          'Blue_Block_Ungrounded_Flat_on_top_of_Red_Block_Grounded_Upright'
        Returns: list of dicts
        """
        tokens = name.split("_")

        # Simple case: just one piece
        if "on_top_of" not in name:
            color, shape, grounded, orientation = tokens[:4]
            if grounded.lower() not in ["grounded", "ungrounded"]:
                grounded = tokens[3]
                orientation = tokens[2]
            return [{
                "color": color,
                "shape": shape,
                "grounded": grounded,
                "orientation": orientation,
            }]

        # Complex case: two stacked pieces
        idx = tokens.index("on")  # start of 'on_top_of'
        top_tokens = tokens[:idx]          # e.g. Blue_Block_Ungrounded_Flat
        bottom_tokens = tokens[idx+3:]     # skip "on_top_of"

        top_color, top_shape, top_grounded, top_orientation = top_tokens[:4]
        if top_grounded.lower() not in ["grounded", "ungrounded"]:
            top_grounded = top_tokens[3]
            top_orientation = top_tokens[2]
        bottom_color, bottom_shape, bottom_grounded, bottom_orientation = bottom_tokens[:4]
        if bottom_grounded.lower() not in ["grounded", "ungrounded"]:
            bottom_grounded = bottom_tokens[3]
            bottom_orientation = bottom_tokens[2]

        return [
            {
                "color": top_color,
                "shape": top_shape,
                "grounded": top_grounded,
                "orientation": top_orientation,
            },
            {
                "color": bottom_color,
                "shape": bottom_shape,
                "grounded": bottom_grounded,
                "orientation": bottom_orientation,
            },
        ]

    def _map_to_ints(self, attrs):
        color_map = {"red": 0, "blue": 1, "yellow": 2}
        shape_map = {"block": 0, "wedge": 1, "pyramid": 2}
        orientation_map = {
            "upright": 0,
            "upsidedown": 1,
            "flat": 2,
            "cheesecake": 3,
            "doorstop": 2,
        }
        color = color_map.get(attrs["color"].lower(), -1)
        shape = shape_map.get(attrs["shape"].lower(), -1)
        orientation = orientation_map.get(attrs["orientation"].lower(), -1)
        return color, shape, orientation

    def _load_labels(self, label_path):
        pieces = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                pieces.append((cls_idx, x, y, w, h))
        return pieces

    def _scene_to_tensor(
        self,
        objects,
        img_path=None,
        *,
        orig_w=4608,
        orig_h=2592,
        cropped_left=1324,
        cropped_top=518,
        cropped_w=2420,
        cropped_h=1815,
        final_w=640,
        final_h=480,
        drop_if_outside=True,
        min_intersect_px=1
    ):
        """
        obj["bbox"] must be normalized (xmin, ymin, xmax, ymax) w.r.t. ORIGINAL image.
        Returns rows with bbox as ABSOLUTE pixel coords (xmin, ymin, xmax, ymax) in FINAL image (e.g. 640x480).
        """
        tensors = []

        crop_right  = cropped_left + cropped_w
        crop_bottom = cropped_top  + cropped_h
        sx = final_w / cropped_w
        sy = final_h / cropped_h

        rows_kept_by_id = {}

        for obj in objects:
            xmin_n, ymin_n, xmax_n, ymax_n = obj["bbox"]

            xmin = xmin_n * orig_w
            ymin = ymin_n * orig_h
            xmax = xmax_n * orig_w
            ymax = ymax_n * orig_h

            # intersect with effective crop (in original pixels)
            ixmin = max(xmin, cropped_left)
            iymin = max(ymin, cropped_top)
            ixmax = min(xmax, crop_right)
            iymax = min(ymax, crop_bottom)

            # check intersection validity
            if ixmax - ixmin < min_intersect_px or iymax - iymin < min_intersect_px:
                if drop_if_outside:
                    continue
                # keep but mark invalid bbox
                relations = [8, 8, 8, 8, 8, 8]
                tensors.append([
                    obj["ID"], obj["color_id"], obj["shape_id"], obj["orientation_id"],
                    *relations, 8,  -1, -1, -1, -1
                ])
                rows_kept_by_id[obj["ID"]] = len(tensors) - 1
                continue

            # shift into cropped coords, then scale into FINAL 640x480 pixels
            x0_final = (ixmin - cropped_left) * sx
            y0_final = (iymin - cropped_top)  * sy
            x1_final = (ixmax - cropped_left) * sx
            y1_final = (iymax - cropped_top)  * sy

            # clamp to valid pixel range
            x0_final = max(0, min(final_w, x0_final))
            y0_final = max(0, min(final_h, y0_final))
            x1_final = max(0, min(final_w, x1_final))
            y1_final = max(0, min(final_h, y1_final))

            relations = [8, 8, 8, 8, 8, 8]  # your placeholders
            row = [
                obj["ID"],
                obj["color_id"],
                obj["shape_id"],
                obj["orientation_id"],
            ] + relations + [8] + [x0_final, y0_final, x1_final, y1_final]

            rows_kept_by_id[obj["ID"]] = len(tensors)
            tensors.append(row)

        # link stacked pieces if both remain
        for obj in objects:
            if "stacked_on" in obj:
                t = rows_kept_by_id.get(obj["ID"])
                b = rows_kept_by_id.get(obj["stacked_on"])
                if t is not None and b is not None:
                    tensors[t][8] = obj["stacked_on"]  # top’s bottom
                    tensors[b][9] = obj["ID"]          # bottom’s top

        # pad/truncate to 7 rows
        while len(tensors) < 7:
            tensors.append([
                self.token_PAD_rel, self.token_PAD, self.token_PAD, self.token_PAD_orientation,
                8, 8, 8, 8, 8, 8, 8,  -1, -1, -1, -1
            ])
        if len(tensors) > 7:
            print(f"Warning: >7 objects ({len(tensors)}) in scene; truncating {img_path}")
            tensors = tensors[:7]

        # optional sanity check: values must be in [0, final_w] x [0, final_h]
        for r in tensors:
            xmin, ymin, xmax, ymax = r[-4:]
            if xmin != -1:
                assert 0 <= xmin <= final_w, f"xmin out of range {xmin} ({img_path})"
                assert 0 <= ymin <= final_h, f"ymin out of range {ymin} ({img_path})"
                assert 0 <= xmax <= final_w, f"xmax out of range {xmax} ({img_path})"
                assert 0 <= ymax <= final_h, f"ymax out of range {ymax} ({img_path})"
                assert xmax >= xmin and ymax >= ymin, f"inverted box {img_path}"

        return torch.tensor(tensors, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path, structure_tensor = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, structure_tensor, str(img_path)
