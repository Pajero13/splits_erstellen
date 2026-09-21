import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = Path("coco")

IMAGE_DIR = BASE_DIR / "val2017"
ANNOTATIONS_FILE = BASE_DIR / "annotations" / "instances_val2017.json"

OUTPUT_DIR = Path("dataset_split")

TOTAL_IMAGES = 1000

TRAIN_SIZE = 800
VAL_SIZE = 100
TEST_SIZE = 100

RANDOM_SEED = 42


# ============================================================
# COCO LADEN
# ============================================================

print("Lade COCO-Annotationen...")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

print(f"COCO-Bilder laut JSON: {len(coco['images'])}")
print(f"COCO-Kategorien: {len(coco['categories'])}")


# ============================================================
# KATEGORIEN
# ============================================================

category_names = {
    category["id"]: category["name"]
    for category in coco["categories"]
}

category_ids = sorted(category_names.keys())


# ============================================================
# BILDER UND KATEGORIEN VERKNÜPFEN
# ============================================================

image_categories = defaultdict(set)

for annotation in coco["annotations"]:

    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    image_categories[image_id].add(category_id)


image_info = {
    image["id"]: image
    for image in coco["images"]
}


# ============================================================
# VORHANDENE BILDER
# ============================================================

available_images = []

for image_id, image in image_info.items():

    image_path = IMAGE_DIR / image["file_name"]

    if image_path.exists():
        available_images.append(image_id)


print(f"Tatsächlich vorhandene Bilder: {len(available_images)}")


# ============================================================
# ZUFALLSSEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# KATEGORIENVERTEILUNG DES GESAMTEN DATENSATZES
# ============================================================

full_category_counts = Counter()

for image_id in available_images:

    for category_id in image_categories[image_id]:
        full_category_counts[category_id] += 1


# ============================================================
# ZIELVERTEILUNG FÜR 1000 BILDER
# ============================================================

target_category_counts = {}

for category_id, count in full_category_counts.items():

    target = (
        count
        / len(available_images)
        * TOTAL_IMAGES
    )

    target_category_counts[category_id] = target


# ============================================================
# 1000 BILDER AUSWÄHLEN
#
# Dieser Teil entspricht unserem ersten Algorithmus.
# Dadurch können wir die beiden Methoden fair vergleichen.
# ============================================================

print("\nWähle 1000 repräsentative Bilder...")

remaining = set(available_images)

selected_images = []

selected_category_counts = Counter()


for iteration in range(TOTAL_IMAGES):

    best_image = None
    best_score = float("-inf")

    candidates = list(remaining)

    random.shuffle(candidates)

    for image_id in candidates:

        categories = image_categories[image_id]

        score = 0.0

        for category_id in categories:

            target = target_category_counts[category_id]

            current = selected_category_counts[category_id]

            if target > 0:

                missing = max(
                    target - current,
                    0
                )

                score += missing / target

        score += len(categories) * 0.001

        if score > best_score:

            best_score = score
            best_image = image_id

    selected_images.append(best_image)

    remaining.remove(best_image)

    for category_id in image_categories[best_image]:
        selected_category_counts[category_id] += 1

    if (iteration + 1) % 100 == 0:
        print(f"  {iteration + 1}/1000")


print(
    f"\nAuswahl abgeschlossen: "
    f"{len(selected_images)} Bilder"
)


# ============================================================
# MULTI-LABEL MATRIX
#
# Jede Zeile = ein Bild
# Jede Spalte = eine COCO-Kategorie
#
# 1 = Kategorie vorhanden
# 0 = Kategorie nicht vorhanden
# ============================================================

print("\nErstelle Multi-Label-Matrix...")

X = np.zeros(
    (TOTAL_IMAGES, len(category_ids)),
    dtype=int
)


category_to_column = {
    category_id: index
    for index, category_id in enumerate(category_ids)
}


for row, image_id in enumerate(selected_images):

    for category_id in image_categories[image_id]:

        column = category_to_column[category_id]

        X[row, column] = 1


# ============================================================
# 80/20 SPLIT
#
# Zuerst:
# 800 Training
# 200 Rest
# ============================================================

print("\nFühre iterative Multilabel-Stratifizierung durch...")

indices = np.arange(TOTAL_IMAGES)


split_1 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=200,
    random_state=RANDOM_SEED
)

train_indices, temp_indices = next(
    split_1.split(indices, X)
)


# ============================================================
# 10/10 SPLIT
#
# Die verbleibenden 200 Bilder werden in
# 100 Validation + 100 Test geteilt.
# ============================================================

temp_X = X[temp_indices]

split_2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=100,
    random_state=RANDOM_SEED
)

val_relative, test_relative = next(
    split_2.split(temp_indices, temp_X)
)


val_indices = temp_indices[val_relative]
test_indices = temp_indices[test_relative]


# ============================================================
# BILD-IDS
# ============================================================

train_ids = [
    selected_images[i]
    for i in train_indices
]

val_ids = [
    selected_images[i]
    for i in val_indices
]

test_ids = [
    selected_images[i]
    for i in test_indices
]


print(f"\nTraining:   {len(train_ids)}")
print(f"Validation: {len(val_ids)}")
print(f"Test:       {len(test_ids)}")


# ============================================================
# DATAFRAME
# ============================================================

def create_dataframe(image_ids, split_name):

    rows = []

    for image_id in image_ids:

        image = image_info[image_id]

        categories = [
            category_names[c]
            for c in sorted(image_categories[image_id])
        ]

        rows.append({
            "image_id": image_id,
            "filename": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "categories": "|".join(categories),
            "split": split_name
        })

    return pd.DataFrame(rows)


train_df = create_dataframe(
    train_ids,
    "train"
)

val_df = create_dataframe(
    val_ids,
    "val"
)

test_df = create_dataframe(
    test_ids,
    "test"
)


all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# ============================================================
# AUSGABEORDNER
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CSV-DATEIEN
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False,
    encoding="utf-8"
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False,
    encoding="utf-8"
)

all_df.to_csv(
    OUTPUT_DIR / "selected_1000.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# TEXTDATEIEN
# ============================================================

for filename, ids in [
    ("train.txt", train_ids),
    ("val.txt", val_ids),
    ("test.txt", test_ids)
]:

    with open(
        OUTPUT_DIR / filename,
        "w",
        encoding="utf-8"
    ) as f:

        for image_id in ids:
            f.write(f"{image_id}\n")


# ============================================================
# ABSCHLUSS
# ============================================================

print("\n========================================")
print("SPLIT ERFOLGREICH ERSTELLT")
print("========================================")

print(f"Gesamt:      {len(all_df)}")
print(f"Training:    {len(train_df)}")
print(f"Validation:  {len(val_df)}")
print(f"Test:        {len(test_df)}")

print("\nMethode:")
print("Iterative Multilabel Stratification")

print("\nRandom Seed:")
print(RANDOM_SEED)

print("\nFertig.")