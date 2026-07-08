import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from zendo_detection.zendo_dataset import ZendoImageToStructureDataset
from zendo_detection.zendo_encoder import ZendoStructureEncoding

root_paths = [
        "../dataset/training2","../dataset/training6", "../dataset/training7", "../dataset/training8",
        "../dataset/training9", "../dataset/training10", "../dataset/training11", "../dataset/training12", 
        "../dataset/training13", "../dataset/training14", 
        "../dataset/training15", "../dataset/training16",
        "../dataset/training17", "../dataset/training18", "../rules-dataset/training19",
        "../rules-dataset/training20", "../rules-dataset/training21", "../rules-dataset/training22",
        "../rules-dataset/training24", "../rules-dataset/training26"
    ]

color_lexicon = ["red", "blue", "yellow"]
shape_lexicon = ["block", "wedge", "pyramid"]
orientation_lexicon = ["upright", "upside_down", "flat", "cheesecake"]
max_objects = 7

encoder = ZendoStructureEncoding(
    color_lexicon=color_lexicon,
    shape_lexicon=shape_lexicon,
    orientation_lexicon=orientation_lexicon,
    max_objects=max_objects
)

dataset = ZendoImageToStructureDataset(
    encoder=encoder,
    root_paths=root_paths,
    transform=None
)

color_counter = Counter()
shape_counter = Counter()
orientation_counter = Counter()
touching_counter = Counter()
pointing_counter = Counter()

total_objects = 0
num_touching = 0
num_pointing = 0
length_counter = Counter()
pad_token_id = encoder.token_PAD

for scene in dataset.scenes:
    structure_tensor = encoder.encode_structure(scene)  # shape: [max_objects, vector_dim]
    length = sum(1 for vec in structure_tensor if vec[1].item() != pad_token_id)
    length_counter[length] += 1
    total_objects += length
    for obj in scene["objects"]:

        # Categorical counts
        color_counter[obj["color"]] += 1
        shape_counter[obj["shape"]] += 1
        orientation_counter[obj["orientation"]] += 1

        # Pointing
        pointing = obj.get("pointing", "")
        if pointing and pointing.strip().lower() not in ["", "none", "null"]:
            num_pointing += 1
            pointing_counter[1] += 1
        else:
            pointing_counter[0] += 1

        # Touching
        touching_rel = obj.get("touching", {})
        touching_valid = any(v not in ["None", None, "null", ""] for v in touching_rel.values())
        touching_counter[int(touching_valid)] += 1
        if touching_valid:
            num_touching += 1

# Summary
pct_pointing = 100.0 * num_pointing / total_objects
pct_touching = 100.0 * num_touching / total_objects

print(f"Total objects: {total_objects}")
print(f"Pointing: {num_pointing} ({pct_pointing:.2f}%)")
print(f"Touching: {num_touching} ({pct_touching:.2f}%)")

def plot_histogram(counter, title, xlabel):
    items = sorted(counter.items())
    labels, values = zip(*items)
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)
    plt.xticks(rotation=45)

plot_histogram(length_counter, "Structure Length Distribution", "Number of Objects per Scene")
plot_histogram(color_counter, "Color Distribution", "Color")
plot_histogram(shape_counter, "Shape Distribution", "Shape")
plot_histogram(orientation_counter, "Orientation Distribution", "Orientation")

plt.figure()
plt.bar(["Not Touching", "Touching"], [touching_counter[0], touching_counter[1]], color="steelblue")
plt.title("Touching Relation Presence")
plt.ylabel("Number of Objects")
plt.grid(axis="y")

plt.figure()
plt.bar(["Not Pointing", "Pointing"], [pointing_counter[0], pointing_counter[1]], color="orange")
plt.title("Pointing Relation Presence")
plt.ylabel("Number of Objects")
plt.grid(axis="y")

# --- Plot Percentage Summary ---
plt.figure(figsize=(6, 4))
labels = ["Touching", "Pointing"]
values = [pct_touching, pct_pointing]
plt.bar(labels, values, color=["steelblue", "orange"])
plt.ylabel("Percentage of Objects (%)")
plt.title("Objects with Touching or Pointing Relations")
plt.ylim(0, 100)
plt.grid(axis="y")

plt.show()