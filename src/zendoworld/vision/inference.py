import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

base_cfg = {
    "max_objects": 7,
    "token_dim": 384,
    "color_lexicon": ["red", "blue", "yellow"],
    "shape_lexicon": ["block", "wedge", "pyramid"],
    "orientation_lexicon": ["upright", "upside_down", "flat", "cheesecake"],
    "dropout": 0.23,
    "layers": 4,
    "pointing_mult_layer": True,
    "touching_mult_layer": True,
    "bbox_mult_layer": True,
    "color_mult_layer": False,
    "shape_mult_layer": False,
    "orientation_mult_layer": False,
    "presence_mult_layer": False,
}

token_NONE = 8

def call_vision_model(visionmodel, image_path, device=None, presence_thresh=0.5):
    print(f"Calling vision model on image: {image_path}")
    if device is None:
        device = next(visionmodel.parameters()).device

    image_transforms = Compose([Resize((480, 640)), ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = visionmodel(image_tensor)

    def cpu(t):
        return t.detach().cpu() if isinstance(t, torch.Tensor) else t

    color_logits = cpu(outputs["color"])[0]
    shape_logits = cpu(outputs["shape"])[0]
    orient_logits = cpu(outputs["orientation"])[0]
    pointing_logits = cpu(outputs["pointing"])[0]
    touching_logits = cpu(outputs["touching"])[0]
    bbox_out = cpu(outputs["bbox"])[0]
    presence_logits = cpu(outputs["presence"])[0]

    colors = color_logits.argmax(dim=-1).tolist()
    shapes = shape_logits.argmax(dim=-1).tolist()
    orients = orient_logits.argmax(dim=-1).tolist()
    pointing = pointing_logits.argmax(dim=-1).tolist()
    touching = touching_logits.argmax(dim=-1).tolist()
    bbox = bbox_out.tolist() if isinstance(bbox_out, torch.Tensor) else bbox_out
    presence = torch.sigmoid(presence_logits).squeeze(-1).tolist()

    valid_ids = [i for i in range(base_cfg["max_objects"]) if presence[i] > presence_thresh]

    scene_vec = []
    for i in valid_ids:
        vec = [i, colors[i], shapes[i], orients[i]]

        for d in range(6):
            tv = touching[i][d]
            tv = tv if (0 <= tv < base_cfg["max_objects"]) else token_NONE
            vec.append(tv)

        pt = pointing[i]
        pt = pt if (0 <= pt < base_cfg["max_objects"]) else token_NONE
        vec.append(pt)

        bb = bbox[i] if bbox[i] is not None else [-1, -1, -1, -1]
        vec.extend(bb)

        scene_vec.append(torch.tensor(vec, dtype=torch.long))

    while len(scene_vec) < base_cfg["max_objects"]:
        pad_vec = torch.tensor(
            [7, 3, 3, 4,  *[7]*6, 7,  -1, -1, -1, -1], dtype=torch.long
        )
        scene_vec.append(pad_vec)

    final_tensor = torch.stack(scene_vec)
    return final_tensor
