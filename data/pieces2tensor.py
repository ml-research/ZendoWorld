from collections import defaultdict, deque
import re
import torch
import random

COLOR_MAP = {"red": 0, "blue": 1, "yellow": 2}
SHAPE_MAP = {"block": 0, "wedge": 1, "pyramid": 2}
ORIENTATION_MAP = {"upright": 0, "upside_down": 1, "flat": 2, "cheesecake": 3, "doorstop": 2}

PAD_VECTOR = torch.tensor([7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7], dtype=torch.long)
MAX_OBJECTS = 7

def prolog_strings_to_tensor(structures):
    converted_items = []
    for items in structures:
        rows = []
        actions = {}
        pieces = []

        for item in items:
            match = re.match(r"item\((\d+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(.+)\)", item)
            if not match:
                match = re.match(r'item\((\d+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(.+)\)', item)
                if not match:
                    continue
            item_id = int(match.group(1))
            color = match.group(2)
            shape = match.group(3)
            orientation = match.group(4)
            action = match.group(5).strip()

            actions[item_id] = action
            piece = {
                'id': item_id,
                'color': color,
                'shape': shape,
                'orientation': orientation,
                'touching': [8] * 6,
                'pointing': 8,
            }
            pieces.append(piece)

        pieces.sort(key=lambda x: x['id'])
        id_to_index = {p['id']: i for i, p in enumerate(pieces)}
        for piece in pieces:
            row = [piece['id']]
            row.append(COLOR_MAP[piece["color"]])
            row.append(SHAPE_MAP[piece["shape"]])

            orientation = piece["orientation"]
            if orientation == "vertical":
                row.append(random.choice([0, 1]))
            elif orientation == "horizontal":
                row.append(random.choice([2, 3]))
            else:
                row.append(ORIENTATION_MAP[orientation])

            row.extend(piece['touching'])
            row.append(piece['pointing'])

            rows.append(torch.tensor(row, dtype=torch.long))

        tensor = torch.stack(rows)
        for src_id, act in actions.items():
            if act.startswith("touching("):
                tgt_id = _safe_parse_id(act, "touching(")
                if tgt_id is None:
                    continue
                if src_id in id_to_index and tgt_id in id_to_index:
                    if src_id not in tensor[id_to_index[tgt_id]][4:10] and tgt_id not in tensor[id_to_index[src_id]][4:10]:
                        direction = random.choice([0, 2])
                        tensor[id_to_index[src_id]][4 + direction] = id_to_index[tgt_id]
                        tensor[id_to_index[tgt_id]][4 + direction + 1] = id_to_index[src_id]
            elif act.startswith("pointing("):
                tgt_id = _safe_parse_id(act, "pointing(")
                if tgt_id is None:
                    continue
                if src_id in id_to_index and tgt_id in id_to_index:
                    tensor[id_to_index[src_id]][10] = id_to_index[tgt_id]
            elif act.startswith("on_top_of("):
                tgt_id = _safe_parse_id(act, "on_top_of(")
                if tgt_id is None:
                    continue
                if src_id in id_to_index and tgt_id in id_to_index:
                    tensor[id_to_index[src_id]][4 + 5] = id_to_index[tgt_id]
                    tensor[id_to_index[tgt_id]][4 + 4] = id_to_index[src_id]

        while tensor.shape[0] < MAX_OBJECTS:
            tensor = torch.cat([tensor, PAD_VECTOR.unsqueeze(0)], dim=0)
        converted_items.append(tensor)

    return converted_items

def _safe_parse_id(act, prefix):
    val = act[len(prefix):-1].strip()
    if val.lower() == "none" or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None