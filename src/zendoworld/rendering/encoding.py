import torch
from pathlib import Path

def convert_to_tensor(data, path):
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
      id_map = {obj[0]: i for i, obj in enumerate(data)}
      vecs = []
      for piece in data:
            vec = []
            [name, touching, pointing, x_min, y_min, x_max, y_max] = piece
            vec.append(id_map[name])
            parts = name.split("_")
            _, shape, color = parts[:3]
            orientation = "_".join(parts[3:]).lower()
            shape = shape.lower()
            vec.append(color_to_idx.get(color, token_PAD))
            vec.append(shape_to_idx.get(shape, token_PAD))
            vec.append(orientation_to_idx.get(orientation, token_PAD_orientation))
            vec.append(id_map[touching.get("left")] if touching.get("left") is not None else token_NONE)
            vec.append(id_map[touching.get("right")] if touching.get("right") is not None else token_NONE)
            vec.append(id_map[touching.get("front")] if touching.get("front") is not None else token_NONE)
            vec.append(id_map[touching.get("back")] if touching.get("back") is not None else token_NONE)
            vec.append(id_map[touching.get("top")] if touching.get("top") is not None else token_NONE)
            vec.append(id_map[touching.get("bottom")] if touching.get("bottom") is not None else token_NONE)
            vec.append(id_map[pointing] if pointing is not None else token_NONE)
            bb_features = [
                x_min, x_max, y_min, y_max
            ]
            vec.extend(bb_features)
            vecs.append(torch.tensor(vec, dtype=torch.long))
      while len(vecs) < max_objects:
            pad_tensor = torch.tensor([
                token_PAD_rel, token_PAD, token_PAD, token_PAD_orientation,
                *[token_PAD_rel] * 6, token_PAD_rel, -1, -1, -1, -1
            ], dtype=torch.long)
            vecs.append(pad_tensor)
      encoded_scene = torch.stack(vecs)
      torch.save(encoded_scene, Path(path))
