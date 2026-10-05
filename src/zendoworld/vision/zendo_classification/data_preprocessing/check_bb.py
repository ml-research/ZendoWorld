import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

def plot_image_with_bboxes_corners(image_path, tensor):
    from PIL import Image
    import matplotlib.pyplot as plt, matplotlib.patches as patches
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.imshow(img)
    for row in tensor:
        xmin, ymin, xmax, ymax = row[-4:].tolist()
        if xmin < 0:  # skip padded
            continue
        x0, y0 = xmin, ymin
        w, h = (xmax - xmin), (ymax - ymin)
        ax.add_patch(patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor="red", facecolor="none"))
    ax.set_title(image_path)
    plt.show()

t = torch.tensor([[  0.0000,   1.0000,   1.0000,   3.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   1.0000,   8.0000,   8.0000,   0.0000, 119.8030, 161.7626,
         322.2731],
        [  1.0000,   0.0000,   0.0000,   1.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   8.0000,   0.0000,   8.0000,   0.0000, 119.8030, 161.7626,
         322.2731],
        [  2.0000,   1.0000,   1.0000,   3.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   3.0000,   8.0000,   8.0000, 204.1640, 119.8017, 352.1441,
         306.4013],
        [  3.0000,   0.0000,   0.0000,   0.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   8.0000,   2.0000,   8.0000, 204.1640, 119.8017, 352.1441,
         306.4013],
        [  4.0000,   1.0000,   1.0000,   3.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   5.0000,   8.0000,   8.0000, 389.2879, 142.2823, 540.4099,
         305.3422],
        [  5.0000,   0.0000,   0.0000,   2.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   8.0000,   4.0000,   8.0000, 389.2879, 142.2823, 540.4099,
         305.3422],
        [  7.0000,   3.0000,   3.0000,   4.0000,   8.0000,   8.0000,   8.0000,
           8.0000,   8.0000,   8.0000,   8.0000,  -1.0000,  -1.0000,  -1.0000,
          -1.0000]])
plot_image_with_bboxes_corners("../../zendo_blocks_object_detection/zendo_yolo_dataset_including_labels/train/images_cropped_asym/100_jpg.rf.2a2435bee0268228fb00f302ae2f3f50.jpg", t)
