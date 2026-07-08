import torch
from pathlib import Path
from .hungarian_loss import permutation_invariant_object_loss
import csv
import copy

def update_frozen_heads(model, phase):
    if phase == "bbox":
        frozen = {"color", "shape", "orientation", "pointing", "touching"}
    elif phase == "attributes":
        frozen = {"bbox", "pointing", "touching"}
    elif phase == "relations_only":
        frozen = {"bbox", "color", "shape", "orientation"}
    else: # "relations"
        frozen = set()

    for head in {"color", "shape", "orientation", "pointing", "touching"}:
        requires_grad = head not in frozen
        if hasattr(model, f"{head}_head"):
            for param in getattr(model, f"{head}_head").parameters():
                param.requires_grad = requires_grad
    return frozen

def write_csv_row(path, data_dict):
    log_fields = ["epoch", "loss", "color", "shape", "orientation", "pointing", "touching", "bbox", "presence"]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(data_dict)

def train_model(model, train_loader, val_loader, optimizer, device, csv_path, num_epochs=10, head_weights=None, scheduler=None):
    model.to(device)

    # Only train model on bounding boxes
    training_phase = "bbox"  # or "attributes", "relations"
    frozen_heads = update_frozen_heads(model, training_phase)

    log_dir = Path(csv_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = log_dir / "train_log.csv"
    val_log_path = log_dir / "val_log.csv"

    # Add early stopping
    rolling_val_losses = []
    rolling_start_epoch = 35
    rolling_window = 10
    best_avg_loss = float("inf")
    best_weights = None

    for epoch in range(num_epochs):

        model.train()
        train_stats = dict(epoch=epoch + 1, loss=0.0, color=0.0, shape=0.0,
                           orientation=0.0, pointing=0.0, touching=0.0, bbox=0.0, presence=0.0)
        total_samples = 0
        num_batches = 0

        for batch in train_loader:
            images, targets, paths = batch
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            total_samples += images.size(0)

            (loss, l_col, l_sha, l_ori, l_point, l_touch, l_bb, l_presence) = \
                permutation_invariant_object_loss(outputs, paths, targets, device, {
                "color": outputs["color"].shape[-1],
                "shape": outputs["shape"].shape[-1],
                "orientation": outputs["orientation"].shape[-1],
                "pointing": outputs["pointing"].shape[-1],
                },
                frozen_heads=frozen_heads, head_weights=head_weights
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            # sum batch means; we'll divide by num_batches
            train_stats["loss"]        += loss.item()
            train_stats["color"]       += l_col.item()
            train_stats["shape"]       += l_sha.item()
            train_stats["orientation"] += l_ori.item()
            train_stats["pointing"]    += l_point.item()
            train_stats["touching"]    += l_touch.item()
            train_stats["bbox"]        += l_bb.item()
            train_stats["presence"]    += l_presence.item()
            num_batches += 1

            del loss, l_col, l_sha, l_ori, l_point, l_touch, l_bb, l_presence
            del outputs, images, targets, paths
            torch.cuda.empty_cache()

        # average by number of batches (not samples)
        if num_batches > 0:
            for k in list(train_stats.keys())[1:]:
                train_stats[k] /= num_batches

        print(f"{epoch}: Train | Loss: {train_stats['loss']:.4f}, "
              f"Color: {train_stats['color']:.4f}, Shape: {train_stats['shape']:.4f}, "
              f"Orientation: {train_stats['orientation']:.4f}, "
              f"Pointing: {train_stats['pointing']:.4f}, Touching: {train_stats['touching']:.4f}, "
              f"Bounding_Box: {train_stats['bbox']:.6f}, Presence: {train_stats['presence']:.4f}")
        write_csv_row(train_log_path, train_stats)


        model.eval()
        val_stats = dict(epoch=epoch + 1, loss=0.0, color=0.0, shape=0.0,
                         orientation=0.0, pointing=0.0, touching=0.0, bbox=0.0, presence=0.0)
        total_samples = 0
        num_batches = 0
        with torch.no_grad():
            for images, targets, paths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                B = images.size(0)
                total_samples += B

                # Run model forward
                outputs = model(images)

                # Compute permutation-invariant losses (without enabling relational if not desired)
                (loss, l_col, l_sha, l_ori, l_point, l_touch, l_bb, l_presence) = \
                    permutation_invariant_object_loss(
                    outputs, paths, targets, device,
                    {
                        "color": outputs["color"].shape[-1],
                        "shape": outputs["shape"].shape[-1],
                        "orientation": outputs["orientation"].shape[-1],
                        "pointing": outputs["pointing"].shape[-1],
                    },
                    frozen_heads=frozen_heads, head_weights=head_weights
                )

                val_stats["loss"]        += loss.item()
                val_stats["color"]       += l_col.item()
                val_stats["shape"]       += l_sha.item()
                val_stats["orientation"] += l_ori.item()
                val_stats["pointing"]    += l_point.item()
                val_stats["touching"]    += l_touch.item()
                val_stats["bbox"]        += l_bb.item()
                val_stats["presence"]    += l_presence.item()
                num_batches += 1

                del loss, l_col, l_sha, l_ori, l_point, l_touch, l_bb, l_presence
                del images, targets, paths
                torch.cuda.empty_cache()

        if num_batches > 0:
            for k in list(val_stats.keys())[1:]:
                val_stats[k] /= num_batches

        print(f"\t{epoch}: Validation | Loss: {val_stats['loss']:.4f}, "
              f"Color: {val_stats['color']:.4f}, Shape: {val_stats['shape']:.4f}, "
              f"Orientation: {val_stats['orientation']:.4f}, "
              f"Pointing: {val_stats['pointing']:.4f}, Touching: {val_stats['touching']:.4f}, "
              f"Bounding_Box: {val_stats['bbox']:.4f}, Presence: {val_stats['presence']:.4f}")
        write_csv_row(val_log_path, val_stats)
        if epoch >= rolling_start_epoch and training_phase == "relations":
            rolling_val_losses.append(val_stats["loss"])
            if len(rolling_val_losses) > rolling_window:
                rolling_val_losses.pop(0)
            rolling_avg = sum(rolling_val_losses) / len(rolling_val_losses)
            if rolling_avg < best_avg_loss:
                best_avg_loss = rolling_avg
                best_weights = copy.deepcopy(model.state_dict())
                print(f"[✓] New best model at epoch {epoch} | Rolling Avg Val Loss: {rolling_avg:.4f}")
        
        if training_phase == "bbox" and val_stats["bbox"] < 40:
            training_phase = "attributes"
            frozen_heads = update_frozen_heads(model, training_phase)
            print("Switched to attribute training phase.")
        elif training_phase == "attributes":
            avg_attr = (val_stats["color"] + val_stats["shape"] + val_stats["orientation"]) / 3
            if avg_attr < 0.6:
                training_phase = "relations_only"
                frozen_heads = update_frozen_heads(model, training_phase)
                print("Switched to relation training phase.")
        elif training_phase == "relations_only":
            if val_stats["touching"] < 0.1:
                training_phase = "relations"
                frozen_heads = update_frozen_heads(model, training_phase)
                print("Switched to full training phase.")

    return best_weights, best_avg_loss
