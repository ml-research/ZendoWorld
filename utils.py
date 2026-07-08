from create_programs_from_string import convert_string_to_dsl
from data.pieces2tensor import prolog_strings_to_tensor
from prompts.zendo.rule_conversion import rule_conversion_prompt
from prompts.zendo.perception import perception_prompt
import re
import ast
import torch

def parse_listed_output(outputs):
    try:
        idx = 1
        res = []
        for output in list(filter(None, outputs.split('\n'))):
            if len(output.split(f'{idx}. ')) > 1:
                res.append(output.split(f'{idx}. ')[1].strip(' \n'))
                idx += 1
        return res
    except:
        print("SOMETHING WRONG", outputs, list(filter(None, outputs.split('\n'))))


def list_to_str(lst):
    res = ''
    for idx, x in enumerate(lst):
        res += f'{idx + 1}. {x}\n'
    return res

def extract_dsl_from_hypothesis(h, prompter, cfg, seed=1):
    prompt = rule_conversion_prompt.format(hs=h)
    response = prompter.prompt_with_text(prompt_text=prompt, seed=seed)
    if response is None or response == "":
        print(f"No response for hypothesis '{h}'")
        return None
    try:
        dsl = response.strip()
        if dsl.startswith("```"):
            dsl = extract_python_block(dsl)
        if dsl == "I don't know":
            return None
        print("Trying to convert DSL:", dsl)
        program = convert_string_to_dsl(dsl, cfg)
        return program
    except Exception as e:
        print(f"Failed to convert DSL for hypothesis '{h}': {e}")
        return None


def extract_list_literal(text):
    match = re.search(r"(\[.*?\])", text, re.DOTALL)
    if match:
        return match.group(1)
    return None
    
def extract_python_block(text):
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def get_example_from_path(path, prompter, seed=1):
    response = prompter.prompt_with_images(
        prompt_text=perception_prompt,
        paths=[str(path)],
        seed=seed,
    )
    if response is None or response == "":
        return None
    print(f"Perception response: {response}")
    items = extract_python_block(response)
    if items is None:
        items = extract_list_literal(response)
        if items is None:
            print(f"Failed to extract items from response: {response}")
            return None
    try:
        parsed = ast.literal_eval(items)
    except Exception as e:
            print("Failed to parse response using ast.literal_eval:", e)
            return None
    if not isinstance(parsed, list):
        print("Parsed response is not a list:", parsed)
        return None
    try:
        tensor = prolog_strings_to_tensor([parsed])[0]
        return tensor
    except Exception as e:
        print(f"Failed to convert response to tensor: {e}")
        return None


class ZendoStructure:
    def __init__(self, pieces):
        self.pieces = pieces

    def __repr__(self):
        return f"ZendoStructure(num_pieces={len(self.pieces)})"


class ZendoPiece:
    def __init__(self,
        color: str,
        shape: str,
        orientation: str,
        touching: list[int],
        on_top_of: int | None,
        pointing: int | None):
        self.color = color
        self.shape = shape
        self.orientation = orientation
        self.touching = touching
        self.on_top_of = on_top_of
        self.pointing = pointing

    def __repr__(self):
        return (f"ZendoPiece(color={self.color}, shape={self.shape}, "
                f"orientation={self.orientation}, touching={self.touching}, "
                f"on_top_of={self.on_top_of}, pointing={self.pointing})")


ID_IDX = 0
COLOR_IDX = 1
SHAPE_IDX = 2
ORIENT_IDX = 3
TOUCH_IDX = slice(4, 10)
ON_TOP_IDX = 9
POINT_IDX = 10

COLOR_MAP = {0: "red", 1: "blue", 2: "yellow"}
SHAPE_MAP = {0: "block", 1: "wedge", 2: "pyramid"}
ORIENT_MAP = {
    0: "upright",
    1: "upside_down",
    2: "flat",
    3: "cheesecake",
}


def tensor_to_zendo_structure(structure_tensor: torch.Tensor) -> ZendoStructure:
    """Convert a 2D tensor (num_pieces, feature_dim) into a ZendoStructure, skipping buffer rows (ID==7)."""
    pieces = []

    num_rows = structure_tensor.shape[0]

    for row_idx in range(num_rows):
        piece_tensor = structure_tensor[row_idx]

        piece_id = piece_tensor[ID_IDX].item()

        if piece_id == 7:
            continue

        color_idx = piece_tensor[COLOR_IDX].item()
        shape_idx = piece_tensor[SHAPE_IDX].item()
        orient_idx = piece_tensor[ORIENT_IDX].item()

        color = COLOR_MAP.get(color_idx, "unknown")
        shape = SHAPE_MAP.get(shape_idx, "unknown")
        orientation = ORIENT_MAP.get(orient_idx, "unknown")

        touching_indices = []
        for idx in piece_tensor[TOUCH_IDX].tolist():
            if (
                isinstance(idx, (int, float))
                and 0 <= int(idx) < 8
                and structure_tensor[int(idx)][ID_IDX].item() not in (7, 8)
            ):
                touching_indices.append(int(idx))

        touching_indices = list(set(touching_indices))

        on_top_idx = piece_tensor[ON_TOP_IDX].item()
        on_top_of = None
        if (
            0 <= on_top_idx < 8
            and structure_tensor[int(on_top_idx)][ID_IDX].item() not in (7, 8)
        ):
            on_top_of = int(on_top_idx)

        point_idx = piece_tensor[POINT_IDX].item()
        pointing = None
        if (
            0 <= point_idx < 8
            and structure_tensor[int(point_idx)][ID_IDX].item() not in (7, 8)
        ):
            pointing = int(point_idx)

        piece = ZendoPiece(
            color=color,
            shape=shape,
            orientation=orientation,
            touching=touching_indices,
            on_top_of=on_top_of,
            pointing=pointing,
        )

        pieces.append(piece)

    return ZendoStructure(pieces)