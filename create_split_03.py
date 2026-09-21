import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


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
# GRUNDPRÜFUNG
# ============================================================

if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != TOTAL_IMAGES:
    raise ValueError(
        "TRAIN_SIZE + VAL_SIZE + TEST_SIZE muss "
        "TOTAL_IMAGES ergeben."
    )


# ============================================================
# COCO-ANNOTATIONEN LADEN
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
# VORHANDENE BILDER ERMITTELN
# ============================================================

available_images = []

for image_id, image in image_info.items():

    image_path = IMAGE_DIR / image["file_name"]

    if image_path.exists():
        available_images.append(image_id)


print(
    f"Tatsächlich vorhandene Bilder: "
    f"{len(available_images)}"
)


if len(available_images) < TOTAL_IMAGES:
    raise ValueError(
        f"Es sind nur {len(available_images)} Bilder vorhanden, "
        f"aber {TOTAL_IMAGES} werden benötigt."
    )


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# KATEGORIENVERTEILUNG DES VOLLSTÄNDIGEN COCO-DATENSATZES
# ============================================================

full_category_counts = Counter()

for image_id in available_images:

    for category_id in image_categories[image_id]:
        full_category_counts[category_id] += 1


# ============================================================
# ZIELVERTEILUNG FÜR DIE 1000 BILDER
#
# Beispiel:
# Wenn "person" in 76 % der vorhandenen Bilder vorkommt,
# soll sie ungefähr auch in 76 % unserer 1000 Bilder
# vorkommen.
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
# GREEDY-ANSATZ
#
# Dies entspricht bewusst unserem bisherigen Verfahren,
# damit wir später den alten und neuen Split fair vergleichen
# können.
# ============================================================

print("\nWähle 1000 repräsentative Bilder...")

remaining = set(available_images)

selected_images = []

selected_category_counts = Counter()


for iteration in range(TOTAL_IMAGES):

    best_image = None
    best_score = float("-inf")

    candidates = list(remaining)

    # Zufällige Reihenfolge bei gleichem Score
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

        # Kleiner Bonus für Bilder mit mehreren Kategorien.
        # Dadurch wird die Multi-Label-Diversität unterstützt.
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
# MULTI-LABEL-MATRIX
#
# Zeile = Bild
# Spalte = COCO-Kategorie
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
# ITERATIVE MULTI-LABEL STRATIFIKATION
#
# 1000 Bilder
#     ↓
# 10 Folds mit jeweils 100 Bildern
#
# Danach:
# 8 Folds  = Training
# 1 Fold   = Validation
# 1 Fold   = Test
# ============================================================

print(
    "\nFühre iterative Multi-Label-Stratifizierung durch..."
)

print("Erstelle 10 Folds mit jeweils 100 Bildern...")


indices = np.arange(TOTAL_IMAGES)

n_splits = 10

mskf = MultilabelStratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_SEED
)


folds = []

for fold_number, (_, test_indices) in enumerate(
    mskf.split(indices, X),
    start=1
):

    folds.append(test_indices)

    print(
        f"  Fold {fold_number}: "
        f"{len(test_indices)} Bilder"
    )


# ============================================================
# PRÜFEN, OB JEDER FOLD GENAU 100 BILDER ENTHÄLT
# ============================================================

for fold_number, fold in enumerate(folds, start=1):

    if len(fold) != 100:

        raise RuntimeError(
            f"Fold {fold_number} enthält "
            f"{len(fold)} statt 100 Bilder."
        )


# ============================================================
# TRAIN / VALIDATION / TEST
#
# Die Auswahl der Folds wird deterministisch gemacht.
#
# Fold 1-8  → Training
# Fold 9    → Validation
# Fold 10   → Test
# ============================================================

train_indices = np.concatenate(
    folds[:8]
)

val_indices = folds[8]

test_indices = folds[9]


# ============================================================
# ABSCHLUSSPRÜFUNG DER GRÖSSEN
# ============================================================

if len(train_indices) != TRAIN_SIZE:
    raise RuntimeError(
        f"Training enthält {len(train_indices)} "
        f"statt {TRAIN_SIZE} Bilder."
    )

if len(val_indices) != VAL_SIZE:
    raise RuntimeError(
        f"Validation enthält {len(val_indices)} "
        f"statt {VAL_SIZE} Bilder."
    )

if len(test_indices) != TEST_SIZE:
    raise RuntimeError(
        f"Test enthält {len(test_indices)} "
        f"statt {TEST_SIZE} Bilder."
    )


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


# ============================================================
# ÜBERSICHT
# ============================================================

print("\n========================================")
print("SPLIT-GRÖSSEN")
print("========================================")

print(
    f"Training:   {len(train_ids)}"
)

print(
    f"Validation: {len(val_ids)}"
)

print(
    f"Test:       {len(test_ids)}"
)

print(
    f"Gesamt:     "
    f"{len(train_ids) + len(val_ids) + len(test_ids)}"
)


# ============================================================
# DATAFRAME ERSTELLEN
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
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)


# ============================================================
# OUTPUT-ORDNER
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
            f.write(
                f"{image_id}\n"
            )


# ============================================================
# ABSCHLUSS
# ============================================================

print("\n========================================")
print("SPLIT ERFOLGREICH ERSTELLT")
print("========================================")

print(
    f"Gesamt:      {len(all_df)}"
)

print(
    f"Training:    {len(train_df)}"
)

print(
    f"Validation:  {len(val_df)}"
)

print(
    f"Test:        {len(test_df)}"
)

print("\nMethode:")
print(
    "Greedy-Auswahl + "
    "Multilabel-Stratified 10-Fold Split"
)

print("\nRandom Seed:")
print(RANDOM_SEED)

print("\nDateien:")

print(
    OUTPUT_DIR / "train.csv"
)

print(
    OUTPUT_DIR / "val.csv"
)

print(
    OUTPUT_DIR / "test.csv"
)

print(
    OUTPUT_DIR / "selected_1000.csv"
)

print(
    OUTPUT_DIR / "train.txt"
)

print(
    OUTPUT_DIR / "val.txt"
)

print(
    OUTPUT_DIR / "test.txt"
)

print("\nFertig.")