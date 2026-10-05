import os
from PIL import Image, ImageOps

def smart_asymmetric_crop_and_resize(
    input_dir,
    output_dir,
    target_size=(640, 480),
    left_frac=0.20,   # fraction of width to remove from LEFT
    right_frac=0.10,  # fraction of width to remove from RIGHT
    top_frac=0.20,    # fraction of height to remove from TOP
    bottom_frac=0.10, # fraction of height to remove from BOTTOM
    jpeg_quality=85,
    verbose=True,
    print_every=10,
    max_prints=20
):
    """
    1) Asymmetric border crop.
    2) Aspect crop to reach target aspect ratio.
    3) Resize to target_size.
    Prints the *effective* crop (relative to the original image) and scale factors.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        RESAMPLE = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE = Image.LANCZOS

    tgt_w, tgt_h = target_size
    tgt_ar = tgt_w / tgt_h

    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"))]
    img_files.sort()

    n_printed = 0
    for i, fname in enumerate(img_files, start=1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        with Image.open(in_path) as im:
            im = ImageOps.exif_transpose(im)
            orig_w, orig_h = im.size
            orig_ar = orig_w / orig_h

            # Asymmetric border crop (relative to original)
            left_px   = int(orig_w * left_frac)
            right_px  = int(orig_w * right_frac)
            top_px    = int(orig_h * top_frac)
            bottom_px = int(orig_h * bottom_frac)

            left   = left_px
            right  = orig_w - right_px
            top    = top_px
            bottom = orig_h - bottom_px

            # guard
            right = max(right, left + 2)
            bottom = max(bottom, top + 2)

            # Apply first crop
            im = im.crop((left, top, right, bottom))
            w, h = im.size
            current_ar = w / h

            # Initialize aspect-crop offsets (relative to first-cropped image)
            left2 = top2 = 0
            right2 = w
            bottom2 = h
            aspect_action = "none"
            removed = 0

            if current_ar > tgt_ar:
                new_w = int(h * tgt_ar)
                left2  = (w - new_w) // 2
                right2 = left2 + new_w
                removed = w - new_w
                aspect_action = "crop_width"
                im = im.crop((left2, 0, right2, h))
            elif current_ar < tgt_ar:
                new_h = int(w / tgt_ar)
                top2    = (h - new_h) // 2
                bottom2 = top2 + new_h
                removed = h - new_h
                aspect_action = "crop_height"
                im = im.crop((0, top2, w, bottom2))
            w2, h2 = im.size

            cropped_left = left + left2
            cropped_top  = top + top2
            cropped_right = (left + right2) if aspect_action == "crop_width" else right
            cropped_bottom = (top + bottom2) if aspect_action == "crop_height" else bottom
            cropped_w = cropped_right - cropped_left
            cropped_h = cropped_bottom - cropped_top

            scale_x = tgt_w / cropped_w
            scale_y = tgt_h / cropped_h

            if verbose and (i % print_every == 1) and (n_printed < max_prints):
                print(f"\n[{i}/{len(img_files)}] {fname}")
                print(f"  Original: {orig_w}x{orig_h}  AR={orig_ar:.4f}")
                print(f"  Asym crop px: L={left_px}, R={right_px}, T={top_px}, B={bottom_px}")
                print(f"  After asym crop: {w}x{h} (AR={w/h:.4f})")
                if aspect_action == "crop_width":
                    print(f"  Aspect fix: {aspect_action} → removed {removed}px width")
                elif aspect_action == "crop_height":
                    print(f"  Aspect fix: {aspect_action} → removed {removed}px height")
                else:
                    print(f"  Aspect fix: none (already {tgt_ar:.4f})")
                print(f"  Effective crop (relative to original): "
                      f"left={cropped_left}, top={cropped_top}, "
                      f"width={cropped_w}, height={cropped_h}")
                print(f"  Scales to {tgt_w}x{tgt_h}: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")

                n_printed += 1

            im = im.resize(target_size, RESAMPLE)

            ext = os.path.splitext(fname)[1].lower()
            if ext in (".jpg", ".jpeg"):
                im.save(out_path, format="JPEG",
                        quality=jpeg_quality, optimize=True,
                        subsampling="4:2:0", progressive=True)
            elif ext == ".png":
                im.save(out_path, format="PNG", optimize=True, compress_level=9)
            else:
                im.save(out_path)

        if not verbose or (n_printed >= max_prints):
            if i % max(1, print_every*5) == 0 or i == len(img_files):
                print(f"Processed {i}/{len(img_files)} images")


# Example usage:
smart_asymmetric_crop_and_resize("../../zendo_blocks_object_detection/zendo_yolo_dataset_including_labels/test/images", "../../zendo_blocks_object_detection/zendo_yolo_dataset_including_labels/test/images_cropped_asym")
