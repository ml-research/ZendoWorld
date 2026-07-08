import torch

REVERSE_COLOR_MAP = {0: "red", 1: "blue", 2: "yellow"}
REVERSE_SHAPE_MAP = {0: "block", 1: "wedge", 2: "pyramid"}
REVERSE_ORIENTATION_MAP = {0: "upright", 1: "upside_down", 2: "flat", 3: "cheesecake", 4: "doorstop"}

PAD_ID = 7
PAD_REL = 8

def tensor_to_prolog_strings(tensor_list):
    all_outputs = []

    for tensor in tensor_list:
        pieces = []
        id_map = {}

        for i, row in enumerate(tensor):
            if row[0].item() != PAD_ID:
                id_map[i] = row[0].item()

        for i, row in enumerate(tensor):
            if row[0].item() == PAD_ID:
                continue

            idx = row[0].item()
            color = REVERSE_COLOR_MAP.get(row[1].item(), "unknown")
            shape = REVERSE_SHAPE_MAP.get(row[2].item(), "unknown")
            orientation = REVERSE_ORIENTATION_MAP.get(row[3].item(), "unknown")
            if shape == "wedge" and orientation == "flat":
                orientation = "doorstop"

            action = "grounded"
            pointing = row[10].item()
            on_top = row[9].item()
            below = row[8].item()
            if pointing in id_map:
                action = f"pointing({id_map[pointing]})"
            elif on_top != PAD_REL:
                if on_top in id_map:
                    action = f"on_top_of({id_map[on_top]})"
            elif below != PAD_REL:
                action = "grounded"
            else:
                for d in range(6):
                    tgt = row[4 + d].item()
                    if tgt in id_map and d % 2 == 0:
                        action = f"touching({id_map[tgt]})"

            pieces.append(f"item({idx}, {color}, {shape}, {orientation}, {action})")

        all_outputs.append(pieces)

    return all_outputs
