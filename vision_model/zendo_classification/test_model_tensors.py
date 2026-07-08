import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import argparse
from pathlib import Path

from zendo_detection.zendo_encoder import ZendoStructureEncoding
from zendo_detection.model import ZendoImageToVectorModel

base_cfg = {
    "path": "",
    "seed": 42,

    # Model
    "max_objects": 7,
    "token_dim": 384,

    # Lexicons
    "color_lexicon": ["red", "blue", "yellow"],
    "shape_lexicon": ["block", "wedge", "pyramid"],
    "orientation_lexicon": ["upright", "upside_down", "flat", "cheesecake"],

    # Data
    "val_percent": 0.1,
    "image_size": [480, 640],
    "batch_size": 16,

    # Augmentations
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.05
    },

    # Optimization
    "lr": 5.8e-5,
    "weight_decay": 7e-3,
    "lr_step_size": 10,
    "lr_gamma": 0.65,
    "num_epochs": 60,
    "dropout": 0.23,

    "weight_path": "zendo_model.pt",
    "scheduler_path": "scheduler.pt",

    # Head weights
    "classification": 0.126,
    "relations": 0.12,
    "bbox": 0.00997,

    "layers": 4,
    "color_mult_layer": False,
    "shape_mult_layer": False,
    "orientation_mult_layer": False,
    "presence_mult_layer": False,
    "pointing_mult_layer": True,
    "touching_mult_layer": True,
    "bbox_mult_layer": True,
}
# ── Config ───────────────────────────────────────────────────────────────────
max_objects = 7
threshold = 0.5

color_lexicon = ["red", "blue", "yellow"]
shape_lexicon = ["block", "wedge", "pyramid"]
orientation_lexicon = ["upright", "upside_down", "flat", "cheesecake"]

# IDs 0..max_objects-1 are object indices, max_objects is "self", 8 is NONE.
token_PAD = len(color_lexicon)
token_PAD_shape = len(shape_lexicon)
token_PAD_orientation = len(orientation_lexicon)
token_PAD_rel = 7
token_NONE = 8

image_transforms = Compose([
    Resize((480, 640)),
    ToTensor(),
])

color_to_idx = {c: i for i, c in enumerate(color_lexicon)}
shape_to_idx = {s: i for i, s in enumerate(shape_lexicon)}
orientation_to_idx = {o: i for i, o in enumerate(orientation_lexicon)}


def predict_all_images(image_dir, model_path):
    image_dir = Path(image_dir)
    output_dir = Path("../pred")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZendoImageToVectorModel(
        base_cfg,
        num_output_tokens=max_objects,
        token_dim=384,
        max_objects=max_objects,
        num_colors=len(color_lexicon) + 1,
        num_shapes=len(shape_lexicon) + 1,
        num_orientations=len(orientation_lexicon) + 1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    print("loaded model")
    model.eval()
    print(f"Processing images in {image_dir}")
    for image_path in image_dir.rglob("*.png"):
        vec = []
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)

            colors = outputs["color"][0].argmax(dim=-1).tolist()
            shapes = outputs["shape"][0].argmax(dim=-1).tolist()
            orients = outputs["orientation"][0].argmax(dim=-1).tolist()
            pointing = outputs["pointing"][0].argmax(dim=-1).tolist()
            touching = outputs["touching"][0].argmax(dim=-1).tolist()  # [T, 6]
            bboxes = outputs["bbox"][0].tolist()  # [T, 4]
            presence = torch.sigmoid(outputs["presence"][0]).squeeze(-1).tolist()  # [T]

            for i in range(max_objects):
                if presence[i] > threshold:
                    obj_vec = torch.tensor([
                        i,
                        colors[i],
                        shapes[i],
                        orients[i],
                        *touching[i],
                        pointing[i] if pointing[i] < max_objects else token_NONE,
                        int(bboxes[i][0]), int(bboxes[i][1]),
                        int(bboxes[i][2]), int(bboxes[i][3])
                    ], dtype=torch.long)
                    vec.append(obj_vec)

            while len(vec) < max_objects:
                pad_tensor = torch.tensor([
                    token_PAD_rel, token_PAD, token_PAD_shape, token_PAD_orientation,
                    *[token_PAD_rel] * 6, token_PAD_rel, -1, -1, -1, -1
                ], dtype=torch.long)
                vec.append(pad_tensor)

            tensor_output = torch.stack(vec)
            pt_path = output_dir / image_path.with_suffix(".pt").name
            torch.save(tensor_output, pt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--model", type=str, default="zendo_model.pt", help="Path to trained model")
    args = parser.parse_args()

    predict_all_images(args.image_dir, args.model)
