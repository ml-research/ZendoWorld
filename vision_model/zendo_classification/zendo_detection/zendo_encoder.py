import torch

class ZendoStructureEncoding:
    def __init__(self, color_lexicon, shape_lexicon, orientation_lexicon, max_objects, bbox_bounds=(0, 640)):
        """
        :param color_lexicon: list of valid color strings
        :param shape_lexicon: list of valid shape strings
        :param orientation_lexicon: list of valid orientation strings
        :param max_objects: max number of pieces in one scene
        :param position_bounds: min and max bounds for object positions
        """
        self.max_objects = max_objects
        self.directions = ["left", "right", "front", "back", "top", "bottom"]
        self.invalid_class_index = -100

        # Add PAD and NONE tokens to each lexicon
        self.color_lexicon = color_lexicon + ["PAD"]
        self.shape_lexicon = shape_lexicon + ["PAD"]
        self.orientation_lexicon = orientation_lexicon + ["PAD"]

        self.color_to_idx = {s: i for i, s in enumerate(self.color_lexicon)}
        self.shape_to_idx = {s: i for i, s in enumerate(self.shape_lexicon)}
        self.orientation_to_idx = {s: i for i, s in enumerate(self.orientation_lexicon)}
        self.orientation_to_idx["doorstop"] = 2

        self.token_PAD = 3
        self.token_PAD_orientation = self.orientation_to_idx.get("PAD", -2)
        self.token_PAD_rel = max_objects
        self.token_NONE = max_objects + 1
        print(self.color_lexicon, self.token_PAD, self.shape_lexicon, self.token_PAD_orientation, self.orientation_lexicon, self.token_NONE)

        self.min_val, self.max_val = bbox_bounds
        self.vector_length = 1 + 1 + 1 + 1 + 6 + 1  # id + color + shape + orientation + touching + pointing
        self.bb_feature_length = 4
        self.output_dimension = (self.vector_length + self.bb_feature_length) * self.max_objects

    def normalize_pos(self, val):
        scale = 2.0 / (self.max_val - self.min_val)
        offset = -1.0 - self.min_val * scale
        return val * scale + offset

    def encode_piece(self, piece, id_map):
        vec = []

        vec.append(id_map.get(piece["ID"], self.token_NONE))
        color = piece.get("color", "PAD")
        vec.append(self.color_to_idx.get(color, self.invalid_class_index))

        shape = piece.get("shape", "PAD")
        vec.append(self.shape_to_idx.get(shape, self.invalid_class_index))

        orientation = piece.get("orientation", "PAD")
        vec.append(self.orientation_to_idx.get(orientation, self.invalid_class_index))

        # Touching (6 directions) as object indices
        for dir in self.directions:
            target_id = piece.get("touching", {}).get(dir, None)
            if target_id and target_id in id_map:
                vec.append(id_map[target_id])
            else:
                vec.append(self.token_NONE)

        # Pointing as index
        pointed_id = piece.get("pointing", "")
        pointed_list = piece.get("pointing", [])
        pointed_id = pointed_list[0] if isinstance(pointed_list, list) and pointed_list else pointed_id
        if pointed_id and pointed_id in id_map:
            vec.append(id_map[pointed_id])
        else:
            vec.append(self.token_NONE)

        # Float features
        bb = piece.get("bbox", {})
        bb_features = [bb.get("x", {}).get("min", -1), bb.get("x", {}).get("max",-1),
                          bb.get("y", {}).get("min", -1), bb.get("y", {}).get("max", -1)]

        return torch.cat([
            torch.tensor(vec, dtype=torch.long),
            torch.tensor(bb_features, dtype=torch.long)
        ])

    def encode_structure(self, structure, gt=True):
        if gt:
            objects = sorted(structure["objects"], key=lambda obj: (
                obj.get("bbox", {}).get("x", {}).get("min", -1),
                obj.get("bbox", {}).get("x", {}).get("max", -1),
                obj.get("bbox", {}).get("y", {}).get("min", -1),
                obj.get("bbox", {}).get("y", {}).get("max", -1)
            ))
        else:
            objects = list(structure["objects"])

        id_map = {obj['ID']: idx for idx, obj in enumerate(objects)}

        result = []
        for i in range(self.max_objects):
            if i < len(objects):
                vec = self.encode_piece(objects[i], id_map)
            else:
                pad_values = [
                    self.token_PAD_rel,          # ID
                    self.token_PAD,              # color
                    self.token_PAD,        # shape
                    self.token_PAD_orientation,  # orientation
                    *[self.token_PAD_rel] * 6,    # touching
                    self.token_PAD_rel           # pointing
                ]
                int_pad = torch.tensor(pad_values, dtype=torch.long)
                bb_pad = torch.full((self.bb_feature_length,), -1, dtype=torch.long)
                vec = torch.cat([int_pad, bb_pad])
            result.append(vec)
        return torch.stack(result)
