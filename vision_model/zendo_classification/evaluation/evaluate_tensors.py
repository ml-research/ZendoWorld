import torch
from scipy.optimize import linear_sum_assignment

def evaluate_tensor_files(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, encoder, scene_name=None):
    PAD_color_shape = encoder.token_PAD
    PAD_orientation = encoder.token_PAD_orientation
    PAD_rel = encoder.token_PAD_rel
    None_token = encoder.token_NONE
    T = encoder.max_objects

    results = []

    # Ground truth
    is_real = gt_tensor[:, 1] != PAD_color_shape
    gt_real = gt_tensor[is_real]
    gt_length = gt_real.shape[0]
    gt_bb = gt_real[:, 11:]

    # Predicted presence mask
    is_present = pred_tensor[:, 1] != PAD_color_shape
    pred_real = pred_tensor[is_present]
    pred_bb = pred_real[:, 11:]
    pred_length = pred_real.shape[0]

    if pred_length > gt_length:
        print(f"Predicted more elements than ground truth (Pred: {pred_length}, GT: {gt_length}) in {scene_name}. Cropping.")
    if pred_length == 0 or gt_length == 0:
        print(f"Skipping empty scene {scene_name}.")
        return [], pred_length

    crop_len = min(pred_length, gt_length)
    pred_real = pred_real[:crop_len]
    gt_real = gt_real[:crop_len]
    pred_bb = pred_real[:, 11:]
    gt_bb = gt_real[:, 11:]

    # Hungarian matching by bounding box distance
    cost = ((pred_real[:, None, 11:] - gt_real[None, :, 11:]) ** 2).sum(dim=-1)
    _, col_ind = linear_sum_assignment(cost.cpu())
    perm = torch.tensor(col_ind, dtype=torch.long)

    # Reorder ground truth
    gt_aligned = gt_real[perm]
    pred_aligned = pred_real

    gt_color = gt_aligned[:, 1]
    gt_shape = gt_aligned[:, 2]
    gt_orient = gt_aligned[:, 3]
    pointing_gt_raw = gt_aligned[:, 10].long()
    touching_gt_raw = gt_aligned[:, 4:10].long()
    bb_gt = gt_aligned[:, 11:]

    pred_color = pred_aligned[:, 1]
    pred_shape = pred_aligned[:, 2]
    pred_orient = pred_aligned[:, 3]
    pred_pointing = pred_aligned[:, 10]
    pred_touching = pred_aligned[:, 4:10]

    id_map = {
        int(gt_aligned[i, 0].item()): i
        for i in range(crop_len)
        if int(gt_aligned[i, 0].item()) not in (PAD_rel, None_token)
    }

    with torch.no_grad():
        pointing_gt = pointing_gt_raw.clone()
        for i, pid in enumerate(pointing_gt_raw):
            if pid.item() not in (PAD_rel, None_token):
                pointing_gt[i] = id_map.get(pid.item(), PAD_rel)

        touching_gt = touching_gt_raw.clone()
        for i in range(crop_len):
            for j in range(6):
                tid = touching_gt_raw[i, j].item()
                if tid not in (PAD_rel, None_token):
                    touching_gt[i, j] = id_map.get(tid, PAD_rel)

    # Field-wise accuracy
    for i in range(crop_len):
        result = {
            "color_correct": int(pred_color[i].item() == gt_color[i].item()),
            "shape_correct": int(pred_shape[i].item() == gt_shape[i].item()),
            "orientation_correct": int(pred_orient[i].item() == gt_orient[i].item()),
            "pointing_correct": int(pred_pointing[i].item() == pointing_gt[i].item()),
            "bbox_correct": int(torch.allclose(pred_bb[i].float(), bb_gt[i].float(), atol=50)),
        }
        touching_match = (pred_touching[i] == touching_gt[i]).float().mean().item()
        result["touching_correct"] = int(touching_match >= 0.83)
        result["touching_partial"] = touching_match
        results.append(result)

    return results, pred_length

def evaluate_tensor_files_without_bbox(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, encoder, scene_name=None):
    PAD_color_shape = encoder.token_PAD
    PAD_orientation = encoder.token_PAD_orientation
    PAD_rel = encoder.token_PAD_rel
    None_token = encoder.token_NONE

    # columns (by your schema)
    IDX_ID = 0
    IDX_COLOR = 1
    IDX_SHAPE = 2
    IDX_ORIENT = 3
    IDX_TOUCH_START = 4
    IDX_TOUCH_END = 10
    IDX_POINTING = 10
    IDX_BBOX_START = 11

    results = []

    is_real = gt_tensor[:, IDX_COLOR] != PAD_color_shape
    gt_real = gt_tensor[is_real]
    gt_length = gt_real.shape[0]

    is_present = pred_tensor[:, IDX_COLOR] != PAD_color_shape
    pred_real = pred_tensor[is_present]
    pred_length = pred_real.shape[0]

    if pred_length > gt_length:
        print(f"Predicted more elements than ground truth (Pred: {pred_length}, GT: {gt_length}) in {scene_name}. Cropping.")
    if pred_length == 0 or gt_length == 0:
        print(f"Skipping empty scene {scene_name}.")
        return [], pred_length

    crop_len = min(pred_length, gt_length)
    pred_real = pred_real[:crop_len]
    gt_real = gt_real[:crop_len]

    # ── Hungarian matching (no bbox) ────────────────────────────────────────
    # Cost components per (pred_i, gt_j):
    #   - unary attribute mismatch (color, shape, orientation), 0 or 1 each
    #   - relation existence mismatch on touching[6] + pointing (XOR)
    #   - soft id disagreement (+0.5) if both relations exist but IDs differ pre-alignment
    # bbox columns (11..14) are deliberately ignored.

    p_color = pred_real[:, IDX_COLOR][:, None]      # [P,1]
    p_shape = pred_real[:, IDX_SHAPE][:, None]
    p_orient = pred_real[:, IDX_ORIENT][:, None]

    g_color = gt_real[:, IDX_COLOR][None, :]        # [1,G]
    g_shape = gt_real[:, IDX_SHAPE][None, :]
    g_orient = gt_real[:, IDX_ORIENT][None, :]

    # Unary attribute mismatch (Hamming over 3 attrs)
    unary_mismatch = (p_color != g_color).float() + (p_shape != g_shape).float() + (p_orient != g_orient).float()

    # Touching relations (6 cols)
    p_touch = pred_real[:, IDX_TOUCH_START:IDX_TOUCH_END][:, None, :]   # [P,1,6]
    g_touch = gt_real[:, IDX_TOUCH_START:IDX_TOUCH_END][None, :, :]     # [1,G,6]

    # Existence masks: True if relation points to a real object id (not PAD/None)
    def rel_exists(x):
        return (x != PAD_rel) & (x != None_token)

    p_touch_exists = rel_exists(p_touch)  # [P,1,6]
    g_touch_exists = rel_exists(g_touch)  # [1,G,6]

    touch_existence_mismatch = (p_touch_exists ^ g_touch_exists).float().sum(dim=-1)  # [P,G], XOR over 6 cols

    # Soft disagreement when both exist but IDs differ (penalize lightly since ids aren't aligned yet)
    touch_both_exist = (p_touch_exists & g_touch_exists)
    touch_id_disagree = (p_touch != g_touch) & touch_both_exist
    touch_id_penalty = touch_id_disagree.float().sum(dim=-1) * 0.5  # [P,G]

    # Pointing relation (single id)
    p_point = pred_real[:, IDX_POINTING][:, None]   # [P,1]
    g_point = gt_real[:, IDX_POINTING][None, :]     # [1,G]

    p_point_exists = rel_exists(p_point)
    g_point_exists = rel_exists(g_point)

    point_existence_mismatch = (p_point_exists ^ g_point_exists).float()  # [P,1] ^ [1,G] -> [P,G]
    point_both_exist = (p_point_exists & g_point_exists)
    point_id_penalty = ((p_point != g_point) & point_both_exist).float() * 0.5  # [P,G]

    # Combine costs
    cost = unary_mismatch + touch_existence_mismatch + touch_id_penalty + point_existence_mismatch + point_id_penalty

    _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    perm = torch.tensor(col_ind, dtype=torch.long, device=pred_real.device)

    gt_aligned = gt_real[perm]
    pred_aligned = pred_real

    gt_color = gt_aligned[:, IDX_COLOR]
    gt_shape = gt_aligned[:, IDX_SHAPE]
    gt_orient = gt_aligned[:, IDX_ORIENT]
    pointing_gt_raw = gt_aligned[:, IDX_POINTING].long()
    touching_gt_raw = gt_aligned[:, IDX_TOUCH_START:IDX_TOUCH_END].long()

    pred_color = pred_aligned[:, IDX_COLOR]
    pred_shape = pred_aligned[:, IDX_SHAPE]
    pred_orient = pred_aligned[:, IDX_ORIENT]
    pred_pointing = pred_aligned[:, IDX_POINTING]
    pred_touching = pred_aligned[:, IDX_TOUCH_START:IDX_TOUCH_END]

    id_map = {}
    for i in range(crop_len):
        raw_id = int(gt_aligned[i, IDX_ID].item())
        if raw_id not in (PAD_rel, None_token):
            id_map[raw_id] = i

    with torch.no_grad():
        pointing_gt = pointing_gt_raw.clone()
        for i, pid in enumerate(pointing_gt_raw):
            if pid.item() not in (PAD_rel, None_token):
                pointing_gt[i] = id_map.get(pid.item(), PAD_rel)

        touching_gt = touching_gt_raw.clone()
        for i in range(crop_len):
            for j in range(6):
                tid = touching_gt_raw[i, j].item()
                if tid not in (PAD_rel, None_token):
                    touching_gt[i, j] = id_map.get(tid, PAD_rel)

    for i in range(crop_len):
        result = {
            "color_correct": int(pred_color[i].item() == gt_color[i].item()),
            "shape_correct": int(pred_shape[i].item() == gt_shape[i].item()),
            "orientation_correct": int(pred_orient[i].item() == gt_orient[i].item()),
            "pointing_correct": int(pred_pointing[i].item() == pointing_gt[i].item()),
        }
        touching_match = (pred_touching[i] == touching_gt[i]).float().mean().item()
        result["touching_correct"] = int(touching_match >= 0.83)
        result["touching_partial"] = touching_match
        results.append(result)

    return results, pred_length