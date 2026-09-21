from pathlib import Path
import shutil
import pandas as pd


# ============================================================
# KONFIGURATION
# ============================================================

# COCO-Datensatz
COCO_IMAGE_DIR = Path("coco") / "val2017"

# Die von 04_create_split.py erzeugten CSV-Dateien
SPLIT_DIR = Path("dataset_split")

# Neuer Datensatz für das Experiment
OUTPUT_DIR = Path("experiment_dataset")


# ============================================================
# DATEIEN
# ============================================================

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"


# ============================================================
# GRUNDPRÜFUNGEN
# ============================================================

print("========================================")
print("DATENSATZ VORBEREITEN")
print("========================================")

print("\nPrüfe Dateien...")


required_files = [
    COCO_IMAGE_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Nicht gefunden: {path}"
        )

    print(
        f"  OK: {path}"
    )


# ============================================================
# CSV-DATEIEN LADEN
# ============================================================

print("\nLade Splits...")

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)


print(
    f"Training:   {len(train_df)}"
)

print(
    f"Validation: {len(val_df)}"
)

print(
    f"Test:       {len(test_df)}"
)


# ============================================================
# ERWARTETE GRÖSSEN PRÜFEN
# ============================================================

if len(train_df) != 800:
    raise ValueError(
        f"Training enthält {len(train_df)} "
        f"statt 800 Bilder."
    )


if len(val_df) != 100:
    raise ValueError(
        f"Validation enthält {len(val_df)} "
        f"statt 100 Bilder."
    )


if len(test_df) != 100:
    raise ValueError(
        f"Test enthält {len(test_df)} "
        f"statt 100 Bilder."
    )


# ============================================================
# ORDNER ERSTELLEN
# ============================================================

print("\nErstelle Ordnerstruktur...")


folders = [
    OUTPUT_DIR / "train" / "real",
    OUTPUT_DIR / "train" / "ai",

    OUTPUT_DIR / "val" / "real",
    OUTPUT_DIR / "val" / "ai",

    OUTPUT_DIR / "test" / "real",
    OUTPUT_DIR / "test" / "ai",
]


for folder in folders:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"  {folder}"
    )


# ============================================================
# BILDER KOPIEREN
# ============================================================

def copy_images(
    dataframe,
    split_name
):

    destination = (
        OUTPUT_DIR
        / split_name
        / "real"
    )

    copied = 0
    missing = []

    print(
        f"\nKopiere {split_name}-Bilder..."
    )

    for _, row in dataframe.iterrows():

        filename = str(
            row["filename"]
        )

        source = (
            COCO_IMAGE_DIR
            / filename
        )

        # Zielname:
        # COCO-Dateiname bleibt erhalten, keine Konvertierung in png-Datei
        destination_filename = Path(filename).name

        target = (
            destination
            / destination_filename
        )

        if not source.exists():

            missing.append(
                str(source)
            )

            continue

        shutil.copy2(
            source,
            target
        )

        copied += 1

        if copied % 100 == 0:

            print(
                f"  {copied}/"
                f"{len(dataframe)}"
            )

    if missing:

        print(
            f"\nWARNUNG: "
            f"{len(missing)} Bilder fehlen!"
        )

        for path in missing[:10]:

            print(
                f"  {path}"
            )

        if len(missing) > 10:

            print(
                f"  ... und "
                f"{len(missing) - 10} weitere"
            )

        raise RuntimeError(
            "Nicht alle Bilder konnten "
            "kopiert werden."
        )

    print(
        f"  Fertig: {copied} Bilder"
    )

    return copied


# ============================================================
# TRAIN / VAL / TEST KOPIEREN
# ============================================================

train_copied = copy_images(
    train_df,
    "train"
)

val_copied = copy_images(
    val_df,
    "val"
)

test_copied = copy_images(
    test_df,
    "test"
)


# ============================================================
# ABSCHLUSSPRÜFUNG
# ============================================================

print("\n========================================")
print("ABSCHLUSSPRÜFUNG")
print("========================================")


expected = {
    "train": 800,
    "val": 100,
    "test": 100
}


actual = {
    "train": train_copied,
    "val": val_copied,
    "test": test_copied
}


all_correct = True


for split in [
    "train",
    "val",
    "test"
]:

    expected_count = expected[split]
    actual_count = actual[split]

    status = (
        "OK"
        if expected_count == actual_count
        else "FEHLER"
    )

    print(
        f"{split:10s}: "
        f"{actual_count:4d} / "
        f"{expected_count:4d} "
        f"[{status}]"
    )

    if expected_count != actual_count:

        all_correct = False


# ============================================================
# ABSCHLUSS
# ============================================================

if not all_correct:

    raise RuntimeError(
        "Die Datensatzgrößen stimmen nicht."
    )


print("\n========================================")
print("DATASET ERFOLGREICH ERSTELLT")
print("========================================")

print("\nOrdner:")

print(
    OUTPUT_DIR / "train" / "real"
)

print(
    OUTPUT_DIR / "train" / "ai"
)

print(
    OUTPUT_DIR / "val" / "real"
)

print(
    OUTPUT_DIR / "val" / "ai"
)

print(
    OUTPUT_DIR / "test" / "real"
)

print(
    OUTPUT_DIR / "test" / "ai"
)

print("\nAktueller Stand:")

print(
    "  Training:    800 REAL"
)

print(
    "  Validation:  100 REAL"
)

print(
    "  Test:        100 REAL"
)

print(
    "\nAI-Ordner sind absichtlich noch leer."
)

print(
    "\nDie COCO-Originaldateien wurden "
    "nicht verändert."
)

print("\nFertig.")