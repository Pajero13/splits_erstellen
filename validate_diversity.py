from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# KONFIGURATION
# ============================================================

SPLIT_DIR = Path("dataset_split")

TRAIN_FILE = SPLIT_DIR / "train.csv"
VAL_FILE = SPLIT_DIR / "val.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

TOLERANCE = 0.05


# ============================================================
# DATEN LADEN
# ============================================================

print("Lade Splits...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)


splits = {
    "Training": train_df,
    "Validation": val_df,
    "Test": test_df
}


# ============================================================
# GRUNDPRÜFUNG
# ============================================================

print("\n========================================")
print("GRUNDPRÜFUNG")
print("========================================")

for name, df in splits.items():

    print(f"{name:12s}: {len(df):4d} Bilder")


total_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print(f"{'Gesamt':12s}: {len(total_df):4d} Bilder")


# ============================================================
# KATEGORIEN AUS CSV AUSLESEN
# ============================================================

def get_category_counts(df):

    counter = Counter()

    for categories in df["categories"]:

        if pd.isna(categories):
            continue

        category_list = categories.split("|")

        for category in category_list:
            counter[category] += 1

    return counter


train_counts = get_category_counts(train_df)
val_counts = get_category_counts(val_df)
test_counts = get_category_counts(test_df)
total_counts = get_category_counts(total_df)


# ============================================================
# KATEGORIEN
# ============================================================

all_categories = sorted(
    set(total_counts.keys())
)


# ============================================================
# VERTEILUNG ANALYSIEREN
# ============================================================

print("\n========================================")
print("KATEGORIEVERTEILUNG")
print("========================================")

print(
    f"{'Kategorie':20s}"
    f"{'Gesamt':>8s}"
    f"{'Train':>8s}"
    f"{'Val':>8s}"
    f"{'Test':>8s}"
    f"{'Train %':>10s}"
    f"{'Val %':>10s}"
    f"{'Test %':>10s}"
)

print("-" * 94)


results = []


for category in all_categories:

    total = total_counts[category]
    train = train_counts[category]
    val = val_counts[category]
    test = test_counts[category]

    total_percentage = total / len(total_df)

    train_percentage = train / len(train_df)
    val_percentage = val / len(val_df)
    test_percentage = test / len(test_df)

    # Abweichung vom Anteil des gesamten Datensatzes
    train_deviation = train_percentage - total_percentage
    val_deviation = val_percentage - total_percentage
    test_deviation = test_percentage - total_percentage

    max_deviation = max(
        abs(train_deviation),
        abs(val_deviation),
        abs(test_deviation)
    )

    results.append({
        "category": category,
        "total": total,
        "train": train,
        "val": val,
        "test": test,
        "train_percentage": train_percentage,
        "val_percentage": val_percentage,
        "test_percentage": test_percentage,
        "max_deviation": max_deviation
    })

    print(
        f"{category:20s}"
        f"{total:8d}"
        f"{train:8d}"
        f"{val:8d}"
        f"{test:8d}"
        f"{train_percentage * 100:9.2f}%"
        f"{val_percentage * 100:9.2f}%"
        f"{test_percentage * 100:9.2f}%"
    )


# ============================================================
# MAXIMALE ABWEICHUNG
# ============================================================

print("\n========================================")
print("DIVERSITÄTSBEWERTUNG")
print("========================================")


max_result = max(
    results,
    key=lambda x: x["max_deviation"]
)


max_deviation = max_result["max_deviation"]


print(
    f"Größte Abweichung: "
    f"{max_deviation * 100:.2f} Prozentpunkte"
)

print(
    f"Kategorie mit größter Abweichung: "
    f"{max_result['category']}"
)


# ============================================================
# KATEGORIEN IN JEDEM SPLIT
# ============================================================

train_categories = set(train_counts.keys())
val_categories = set(val_counts.keys())
test_categories = set(test_counts.keys())

print("\n----------------------------------------")

print(
    f"Kategorien im Training:   "
    f"{len(train_categories)}/80"
)

print(
    f"Kategorien in Validation: "
    f"{len(val_categories)}/80"
)

print(
    f"Kategorien im Test:       "
    f"{len(test_categories)}/80"
)


# Fehlende Kategorien
print("\nFehlende Kategorien:")

print(
    "Training:",
    sorted(
        set(all_categories) - train_categories
    )
)

print(
    "Validation:",
    sorted(
        set(all_categories) - val_categories
    )
)

print(
    "Test:",
    sorted(
        set(all_categories) - test_categories
    )
)


# ============================================================
# BEWERTUNG
# ============================================================

print("\n========================================")
print("ERGEBNIS")
print("========================================")


if (
    max_deviation <= TOLERANCE
    and len(train_categories) == 80
    and len(val_categories) == 80
    and len(test_categories) == 80
):

    print(
        "✓ Die Kategorieverteilung ist ausreichend "
        "gleichmäßig."
    )

    print(
        f"✓ Alle 80 Kategorien sind in allen Splits vertreten."
    )

    print(
        f"✓ Maximale Abweichung liegt unter "
        f"{TOLERANCE * 100:.1f} Prozentpunkten."
    )

else:

    print(
        "⚠ Die Kategorieverteilung weist "
        "größere Abweichungen auf."
    )

    print(
        "Der Split sollte möglicherweise "
        "noch optimiert werden."
    )


# ============================================================
# DETAILLIERTE ERGEBNISSE SPEICHERN
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "max_deviation",
    ascending=False
)

results_df.to_csv(
    SPLIT_DIR / "diversity_report.csv",
    index=False,
    encoding="utf-8"
)


print("\nDetailbericht gespeichert unter:")

print(
    SPLIT_DIR / "diversity_report.csv"
)

print("\nFertig.")