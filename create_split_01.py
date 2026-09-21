import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


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


# ============================================================
# BILDER UND KATEGORIEN VERKNÜPFEN
# ============================================================

image_categories = defaultdict(set)

for annotation in coco["annotations"]:

    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    image_categories[image_id].add(category_id)


# Informationen über jedes Bild
image_info = {
    image["id"]: image
    for image in coco["images"]
}


# Nur Bilder verwenden, deren Datei tatsächlich vorhanden ist
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


# ============================================================
# KATEGORIENVERTEILUNG DES GESAMTEN VAL-SPLITS
# ============================================================

full_category_counts = Counter()

for image_id in available_images:

    for category_id in image_categories[image_id]:
        full_category_counts[category_id] += 1


# ============================================================
# 1'000 BILDER AUSWÄHLEN
#
# Wir wollen eine Stichprobe, deren Kategorien möglichst
# ähnlich zur Verteilung des gesamten Val2017-Splits sind.
# ============================================================

print("\nWähle 1000 repräsentative Bilder...")


# Für jede Kategorie bestimmen wir, wie viele Bilder
# ungefähr in unserer 1000er-Stichprobe vorkommen sollen.

target_category_counts = {}

for category_id, count in full_category_counts.items():

    target = count / len(available_images) * TOTAL_IMAGES

    target_category_counts[category_id] = target


# ------------------------------------------------------------
# Greedy-Auswahl
# ------------------------------------------------------------

remaining = set(available_images)
selected_images = []
selected_category_counts = Counter()


for iteration in range(TOTAL_IMAGES):

    best_image = None
    best_score = float("-inf")

    candidates = list(remaining)

    # Zufällige Reihenfolge verhindert deterministische
    # Auswahl bei exakt gleichen Scores.
    random.shuffle(candidates)

    for image_id in candidates:

        categories = image_categories[image_id]

        score = 0.0

        for category_id in categories:

            target = target_category_counts[category_id]
            current = selected_category_counts[category_id]

            if target > 0:

                missing = max(target - current, 0)

                score += missing / target

        # Kleine Belohnung für Bilder mit mehreren Kategorien.
        # Dadurch erhalten wir vielfältige Szenen.
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


print(f"\nAuswahl abgeschlossen: {len(selected_images)} Bilder")


# ============================================================
# 80/10/10 SPLIT
# ============================================================

print("\nErstelle 80/10/10 Split...")

random.shuffle(selected_images)

train_ids = selected_images[:TRAIN_SIZE]

val_ids = selected_images[
    TRAIN_SIZE:
    TRAIN_SIZE + VAL_SIZE
]

test_ids = selected_images[
    TRAIN_SIZE + VAL_SIZE:
]


print(f"Training:   {len(train_ids)}")
print(f"Validation: {len(val_ids)}")
print(f"Test:       {len(test_ids)}")


# ============================================================
# DATAFRAME ERSTELLEN
# ============================================================

def create_dataframe(image_ids, split_name):

    rows = []

    for image_id in image_ids:

        image = image_info[image_id]

        categories = [
            category_names[category_id]
            for category_id in sorted(image_categories[image_id])
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


train_df = create_dataframe(train_ids, "train")
val_df = create_dataframe(val_ids, "val")
test_df = create_dataframe(test_ids, "test")


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
# TEXTDATEIEN MIT BILD-IDS
# ============================================================

with open(
    OUTPUT_DIR / "train.txt",
    "w",
    encoding="utf-8"
) as f:

    for image_id in train_ids:
        f.write(f"{image_id}\n")


with open(
    OUTPUT_DIR / "val.txt",
    "w",
    encoding="utf-8"
) as f:

    for image_id in val_ids:
        f.write(f"{image_id}\n")


with open(
    OUTPUT_DIR / "test.txt",
    "w",
    encoding="utf-8"
) as f:

    for image_id in test_ids:
        f.write(f"{image_id}\n")


# ============================================================
# ZUSAMMENFASSUNG
# ============================================================

print("\n========================================")
print("SPLIT ERFOLGREICH ERSTELLT")
print("========================================")

print(f"Gesamt:      {len(all_df)}")
print(f"Training:    {len(train_df)}")
print(f"Validation:  {len(val_df)}")
print(f"Test:        {len(test_df)}")

print("\nDateien:")

print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "val.csv")
print(OUTPUT_DIR / "test.csv")
print(OUTPUT_DIR / "selected_1000.csv")
print(OUTPUT_DIR / "train.txt")
print(OUTPUT_DIR / "val.txt")
print(OUTPUT_DIR / "test.txt")

print("\nRandom Seed:", RANDOM_SEED)

print("\nFertig.")