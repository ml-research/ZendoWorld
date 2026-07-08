#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import torch

def iter_json_records(p: Path):
    """Yield JSON objects from either .jsonl (one per line) or a single .json file."""
    try:
        text = p.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[WARN] Could not read {p}: {e}", file=sys.stderr)
        return

    # Try single JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            yield obj
            return
        elif isinstance(obj, list):
            # If it's a list of records, yield them
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec
            return
    except Exception:
        pass

    # Fall back to JSON Lines
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                yield rec
            else:
                print(f"[WARN] {p}:{i} not an object; skipping.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}:{i} as JSON: {e}", file=sys.stderr)


def to_torch_tensor(x):
    """Convert nested lists/numbers to a torch.Tensor on CPU."""
    # If it's already a tensor, detach->cpu
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    # If it’s a numpy array or list/scalar, let torch handle it
    try:
        return torch.tensor(x)
    except Exception as e:
        raise TypeError(f"Cannot convert to torch.Tensor: {e}")


def main():
    ap = argparse.ArgumentParser(description="Convert GPT JSON records to .pt tensors.")
    ap.add_argument("--src", type=str, default="gpt_vision", help="Directory with JSON/JSONL files.")
    ap.add_argument("--dst", type=str, default="gpt_predictions", help="Output directory for .pt files.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files.")
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"[ERROR] Source directory does not exist: {src_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(list(src_dir.rglob("*.json"))) + sorted(list(src_dir.rglob("*.jsonl")))
    if not files:
        print(f"[WARN] No .json/.jsonl files found in {src_dir}", file=sys.stderr)

    seen = 0
    written = 0
    for jf in files:
        for rec in iter_json_records(jf):
            seen += 1

            image_path = rec.get("image_path")
            tensor_payload = rec.get("tensor")

            if image_path is None or tensor_payload is None:
                print(f"[WARN] {jf} missing 'image_path' or 'tensor'; skipping record.", file=sys.stderr)
                continue

            try:
                tensor = to_torch_tensor(tensor_payload)
            except Exception as e:
                print(f"[WARN] Failed to convert tensor for {image_path}: {e}", file=sys.stderr)
                continue

            out_name = Path(image_path).stem + ".pt"
            out_path = dst_dir / out_name

            if out_path.exists() and not args.overwrite:
                print(f"[INFO] Skipping existing {out_path} (use --overwrite to replace).", file=sys.stderr)
                continue

            try:
                torch.save(tensor, out_path)
                written += 1
            except Exception as e:
                print(f"[WARN] Failed to save {out_path}: {e}", file=sys.stderr)

    print(f"[DONE] Processed records: {seen}. Saved tensors: {written}. Output dir: {dst_dir}")

if __name__ == "__main__":
    main()
