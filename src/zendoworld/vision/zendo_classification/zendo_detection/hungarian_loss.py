import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def compute_permutation(cost_matrix):
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return torch.tensor(col_ind, dtype=torch.long)

def safe_cross_entropy(logits, targets, ignore_index):
    if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        print("Invalid class index in targets!")
        print("Targets min/max:", targets.min().item(), targets.max().item())
        print("Expected classes:", logits.shape[-1])
    mask = targets != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def permutation_invariant_object_loss(outputs, paths, targets, device, num_classes_dict, frozen_heads=None, head_weights=None):
    """
    outputs: model prediction dict with keys: color, shape, orientation, vector, touching, pointing
    targets: ground truth tensor of shape [B, T, 17]
    num_classes_dict: {'color': C1, 'shape': C2, 'orientation': C3, 'pointing': T+1}
    """
    if head_weights is None:
        head_weights = {
            "classification": 1.0,
            "relations": 1.5,
            "bbox": 0.002,
        }
    bbox_weight       = head_weights["bbox"]
    color_weight      = head_weights["classification"]
    shape_weight      = head_weights["classification"]
    orientation_weight= head_weights["classification"]
    pointing_weight   = head_weights["relations"]
    touching_weight   = head_weights["relations"]
    presence_weight   = head_weights["classification"]

    B, T, D = targets.shape
    total_loss = 0.0
    total_color_loss = 0.0
    total_shape_loss = 0.0
    total_orient_loss = 0.0
    total_point_loss = 0.0
    total_touch_loss = 0.0
    total_bb_loss = 0.0
    total_presence_loss = 0.0

    PAD_color_shape = 3
    PAD_rel = 7
    None_token = 8

    if frozen_heads is None:
        frozen_heads = set()

    for b in range(B):
        pred_bb = outputs["bbox"][b]                     # [T, D]
        presence_logits = outputs["presence"][b].squeeze(-1)

        gt_vector = targets[b]
        is_real = (gt_vector[:, 1] != PAD_color_shape)
        gt_length = is_real.sum().item()

        if gt_length > T:
            print("Ground truth length exceeds or matches T:", gt_length, paths[b])

        gt_color = gt_vector[:, 1]
        gt_shape = gt_vector[:, 2]
        gt_orient = gt_vector[:, 3]
        gt_bb = gt_vector[:, 11:]

        bb_diff = ((pred_bb[:, None, :] - gt_bb[is_real][None, :, :]) ** 2).sum(dim=-1)

        cost_matrix = bb_diff
        perm = compute_permutation(cost_matrix)

        permuted_gt = gt_vector[is_real][perm]        # [valid_T, D]
        padded_gt = gt_vector.clone()
        padded_gt[:gt_length] = permuted_gt

        gt_color = padded_gt[:, 1]
        gt_shape = padded_gt[:, 2]
        gt_orient = padded_gt[:, 3]
        pointing_gt_raw = padded_gt[:, 10].long()
        touching_gt_raw = padded_gt[:, 4:10].long()
        bb_gt = padded_gt[:, 11:]                 # reorder full vector

        permuted_gt_ids = padded_gt[:, 0].long()
        id_map = {
            int(permuted_gt_ids[i]): i
            for i in range(gt_length)
            if int(permuted_gt_ids[i]) not in (PAD_rel, None_token)
        }

        with torch.no_grad():
            pointing_gt = pointing_gt_raw.clone()
            for i, pid in enumerate(pointing_gt_raw):
                if pid.item() not in (PAD_rel, None_token):
                    pointing_gt[i] = id_map.get(pid.item(), PAD_rel)

            touching_gt = touching_gt_raw.clone()
            for i in range(gt_length):
                for j in range(6):
                    tid = touching_gt_raw[i, j].item()
                    if tid not in (PAD_rel, None_token):
                        touching_gt[i, j] = id_map.get(tid, PAD_rel)

        presence_labels = torch.zeros(T, device=device)
        presence_labels[:gt_length] = 1.0

        pred_color_logits = outputs["color"][b]
        pred_shape_logits = outputs["shape"][b]
        pred_orient_logits = outputs["orientation"][b]
        point_logits = outputs["pointing"][b]
        touch_logits = outputs["touching"][b]

        loss_presence = torch.tensor(0.0, device=device) if "presence" in frozen_heads else \
            F.binary_cross_entropy_with_logits(presence_logits, presence_labels)

        loss_color = torch.tensor(0.0, device=device) if "color" in frozen_heads else (
            (presence_labels * F.cross_entropy(pred_color_logits, gt_color.long(), reduction='none')).sum()
            / presence_labels.sum().clamp(min=1.0)
        )

        loss_shape = torch.tensor(0.0, device=device) if "shape" in frozen_heads else (
            (presence_labels * F.cross_entropy(pred_shape_logits, gt_shape.long(), reduction='none')).sum()
            / presence_labels.sum().clamp(min=1.0)
        )

        loss_orient = torch.tensor(0.0, device=device) if "orientation" in frozen_heads else (
            (presence_labels * F.cross_entropy(pred_orient_logits, gt_orient.long(), reduction='none')).sum()
            / presence_labels.sum().clamp(min=1.0)
        )

        loss_bb = F.smooth_l1_loss(pred_bb[:gt_length], bb_gt[:gt_length])

        loss_point = torch.tensor(0.0, device=device) if "pointing" in frozen_heads else (
            (presence_labels * F.cross_entropy(point_logits, pointing_gt.long(), reduction='none')).sum()
            / presence_labels.sum().clamp(min=1.0)
        )

        loss_touch = torch.tensor(0.0, device=device) if "touching" in frozen_heads else F.cross_entropy(
            touch_logits.view(-1, num_classes_dict["pointing"]),
            touching_gt.view(-1),
            ignore_index=-1
        )


        total = (
            color_weight * loss_color +
            shape_weight * loss_shape +
            orientation_weight * loss_orient +
            pointing_weight * loss_point +
            touching_weight * loss_touch +
            bbox_weight * loss_bb +
            presence_weight * loss_presence
        )
        total_loss += total
        total_color_loss += loss_color
        total_shape_loss += loss_shape
        total_orient_loss += loss_orient
        total_point_loss += loss_point
        total_touch_loss += loss_touch
        total_bb_loss += loss_bb
        total_presence_loss += loss_presence

    return total_loss / B, total_color_loss / B, total_shape_loss / B, total_orient_loss / B, total_point_loss / B, total_touch_loss / B, total_bb_loss / B, total_presence_loss / B
