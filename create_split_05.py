import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np


# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = Path("coco")

IMAGE_DIR = BASE_DIR / "train2017"
ANNOTATIONS_FILE = BASE_DIR / "annotations" / "instances_train2017.json"

DATASET_NAME = "dataset_6"  # <- bei jedem Durchlauf anpassen: dataset_1, dataset_2, dataset_3

OUTPUT_DIR = Path(f"dataset_split_{DATASET_NAME}")

USED_IDS_FILE = Path("used_image_ids.txt")

TOTAL_IMAGES = 1000

TRAIN_SIZE = 800
VAL_SIZE = 100
TEST_SIZE = 100

RANDOM_SEED = 42

# Anzahl Optimierungsdurchläufe
OPTIMIZATION_ITERATIONS = 200000


# ============================================================
# GRUNDPRÜFUNGEN
# ============================================================

if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != TOTAL_IMAGES:
    raise ValueError(
        "TRAIN_SIZE + VAL_SIZE + TEST_SIZE muss "
        "TOTAL_IMAGES ergeben."
    )


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
# BILDER UND KATEGORIEN
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


print(
    f"Tatsächlich vorhandene Bilder: "
    f"{len(available_images)}"
)


if len(available_images) < TOTAL_IMAGES:
    raise ValueError(
        f"Es sind nur {len(available_images)} Bilder vorhanden."
    )


# ============================================================
# BEREITS VERWENDETE BILDER AUSSCHLIESSEN
# ============================================================

bereits_verwendet = set()

if USED_IDS_FILE.exists():
    with open(USED_IDS_FILE, "r", encoding="utf-8") as f:
        bereits_verwendet = {int(zeile.strip()) for zeile in f if zeile.strip()}

available_images = [
    img_id for img_id in available_images
    if img_id not in bereits_verwendet
]

print(f"Bereits in anderen Datensätzen verwendet: {len(bereits_verwendet)}")
print(f"Für diesen Datensatz verfügbar: {len(available_images)}")

if len(available_images) < TOTAL_IMAGES:
    raise ValueError(
        f"Nach Ausschluss bereits verwendeter Bilder sind nur "
        f"{len(available_images)} statt {TOTAL_IMAGES} verfügbar."
    )


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# KATEGORIENVERTEILUNG
# ============================================================

full_category_counts = Counter()

for image_id in available_images:

    for category_id in image_categories[image_id]:

        full_category_counts[category_id] += 1


# ============================================================
# ZIELWERTE FÜR 1000 BILDER
# ============================================================

target_category_counts = {}

for category_id, count in full_category_counts.items():

    target_category_counts[category_id] = (
        count
        / len(available_images)
        * TOTAL_IMAGES
    )


# ============================================================
# 1000 BILDER AUSWÄHLEN
#
# IDENTISCH ZUM BISHERIGEN GREEDY-ANSATZ
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

        print(
            f"  {iteration + 1}/1000"
        )


print(
    f"\nAuswahl abgeschlossen: "
    f"{len(selected_images)} Bilder"
)


# ============================================================
# MULTI-LABEL-MATRIX
# ============================================================

print("\nErstelle Multi-Label-Matrix...")

category_to_column = {
    category_id: index
    for index, category_id in enumerate(category_ids)
}


X = np.zeros(
    (TOTAL_IMAGES, len(category_ids)),
    dtype=np.int8
)


for row, image_id in enumerate(selected_images):

    for category_id in image_categories[image_id]:

        column = category_to_column[category_id]

        X[row, column] = 1


# ============================================================
# ZIELWERTE FÜR DIE DREI SPLITS
# ============================================================

# Anzahl der Bilder mit einer Kategorie im 1000er-Datensatz
selected_category_totals = X.sum(axis=0)


# Erwartete Anzahl pro Split
train_targets = selected_category_totals * (
    TRAIN_SIZE / TOTAL_IMAGES
)

val_targets = selected_category_totals * (
    VAL_SIZE / TOTAL_IMAGES
)

test_targets = selected_category_totals * (
    TEST_SIZE / TOTAL_IMAGES
)


# ============================================================
# SCORE-FUNKTION
#
# Wir messen, wie stark die tatsächliche Verteilung
# von der idealen proportionalen Verteilung abweicht.
# ============================================================

def calculate_score(
    train_counts,
    val_counts,
    test_counts,
    train_size,
    val_size,
    test_size
):

    train_expected = (
        selected_category_totals
        * train_size
        / TOTAL_IMAGES
    )

    val_expected = (
        selected_category_totals
        * val_size
        / TOTAL_IMAGES
    )

    test_expected = (
        selected_category_totals
        * test_size
        / TOTAL_IMAGES
    )

    train_error = np.abs(
        train_counts - train_expected
    ).sum()

    val_error = np.abs(
        val_counts - val_expected
    ).sum()

    test_error = np.abs(
        test_counts - test_expected
    ).sum()

    return (
        train_error
        + val_error
        + test_error
    )


# ============================================================
# INITIALER SPLIT
#
# Zunächst zufällige, exakt große Splits.
# ============================================================

print("\nErstelle initialen 800/100/100 Split...")

all_indices = np.arange(TOTAL_IMAGES)

np.random.shuffle(all_indices)


train_indices = all_indices[:TRAIN_SIZE].copy()

val_indices = all_indices[
    TRAIN_SIZE:
    TRAIN_SIZE + VAL_SIZE
].copy()

test_indices = all_indices[
    TRAIN_SIZE + VAL_SIZE:
].copy()


# ============================================================
# COUNT MATRIZEN
# ============================================================

train_counts = X[train_indices].sum(axis=0)

val_counts = X[val_indices].sum(axis=0)

test_counts = X[test_indices].sum(axis=0)


current_score = calculate_score(
    train_counts,
    val_counts,
    test_counts,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE
)


print(
    f"Initialer Score: "
    f"{current_score:.2f}"
)


# ============================================================
# OPTIMIERUNG
#
# Wir tauschen jeweils ein Bild aus zwei verschiedenen Splits.
#
# Der Tausch wird nur akzeptiert, wenn sich die
# Kategorieverteilung verbessert.
# ============================================================

print(
    "\nOptimiere Kategorieverteilung..."
)

split_arrays = [
    train_indices,
    val_indices,
    test_indices
]

split_counts = [
    train_counts,
    val_counts,
    test_counts
]


for iteration in range(
    OPTIMIZATION_ITERATIONS
):

    # Zwei unterschiedliche Splits auswählen
    split_a, split_b = random.sample(
        range(3),
        2
    )

    array_a = split_arrays[split_a]
    array_b = split_arrays[split_b]

    # Zufällige Bilder
    pos_a = random.randrange(
        len(array_a)
    )

    pos_b = random.randrange(
        len(array_b)
    )

    image_a = array_a[pos_a]
    image_b = array_b[pos_b]

    categories_a = X[image_a]
    categories_b = X[image_b]

    # Neue Counts nach Tausch
    new_counts = [
        split_counts[0].copy(),
        split_counts[1].copy(),
        split_counts[2].copy()
    ]

    new_counts[split_a] -= categories_a
    new_counts[split_a] += categories_b

    new_counts[split_b] -= categories_b
    new_counts[split_b] += categories_a

    new_score = calculate_score(
        new_counts[0],
        new_counts[1],
        new_counts[2],
        TRAIN_SIZE,
        VAL_SIZE,
        TEST_SIZE
    )

    # Nur bessere Lösungen akzeptieren
    if new_score < current_score:

        array_a[pos_a] = image_b
        array_b[pos_b] = image_a

        split_counts = new_counts

        current_score = new_score

    if (iteration + 1) % 10000 == 0:

        print(
            f"  {iteration + 1}/"
            f"{OPTIMIZATION_ITERATIONS} "
            f"| Score: {current_score:.2f}"
        )


# ============================================================
# FINALE SPLITS
# ============================================================

train_indices = split_arrays[0]
val_indices = split_arrays[1]
test_indices = split_arrays[2]


# ============================================================
# PRÜFUNG
# ============================================================

if len(train_indices) != 800:
    raise RuntimeError(
        "Training enthält nicht exakt 800 Bilder."
    )

if len(val_indices) != 100:
    raise RuntimeError(
        "Validation enthält nicht exakt 100 Bilder."
    )

if len(test_indices) != 100:
    raise RuntimeError(
        "Test enthält nicht exakt 100 Bilder."
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
# DATAFRAME
# ============================================================

def create_dataframe(
    image_ids,
    split_name
):

    rows = []

    for image_id in image_ids:

        image = image_info[image_id]

        categories = [
            category_names[c]
            for c in sorted(
                image_categories[image_id]
            )
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
# OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# VERWENDETE IDS SPEICHERN (für nächsten Datensatz)
# ============================================================

with open(USED_IDS_FILE, "a", encoding="utf-8") as f:
    for image_id in selected_images:
        f.write(f"{image_id}\n")

print(f"\n{len(selected_images)} IDs zu {USED_IDS_FILE} hinzugefügt.")


# ============================================================
# ABSCHLUSS
# ============================================================

print("\n========================================")
print("SPLIT ERFOLGREICH ERSTELLT")
print("========================================")

print(
    f"Datensatz:   {DATASET_NAME}"
)

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
    "exakte 80/10/10 Split-Optimierung"
)

print(
    "\nOptimierungsiterationen:"
)

print(
    OPTIMIZATION_ITERATIONS
)

print(
    "\nFinaler Optimierungsscore:"
)

print(
    f"{current_score:.2f}"
)

print(
    "\nRandom Seed:"
)

print(
    RANDOM_SEED
)

print("\nFertig.")