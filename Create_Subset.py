from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Konfiguration (hier anpassen)
# ============================================================

INPUT_PATH = Path("data/raw/sales_train_evaluation.csv")
OUTPUT_DIR = Path("data/preprocessed/subsets")
OUTPUT_NAME = "subset_200_series.csv"

TARGET_NUMBER_OF_SERIES = 200
RANDOM_SEED = 42

# Nachfrageklassen über Anteil aktiver Tage (y > 0)
LOW_THRESHOLD = 0.30   # <= 30% aktive Tage
MID_THRESHOLD = 0.70   # 31-70% aktive Tage, Rest => high

# Zielanteile pro Nachfrageklasse (für Modellvergleich)
DEMAND_CLASS_SHARES = {
    "low": 0.25,
    "mid": 0.50,
    "high": 0.25,
}


# ============================================================
# Hilfsfunktionen
# ============================================================

def load_wide_data(input_path: Path) -> pd.DataFrame:
    """Lädt Wide-Format Daten aus CSV (sales_train_evaluation.csv)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {input_path}")

    wide_data = pd.read_csv(input_path)

    required_columns = {"id", "state_id", "cat_id"}
    missing_columns = required_columns - set(wide_data.columns)
    if missing_columns:
        raise ValueError(f"Fehlende Spalten: {sorted(missing_columns)}")

    day_columns = [c for c in wide_data.columns if c.startswith("d_")]
    if not day_columns:
        raise ValueError("Keine Tages-Spalten gefunden (erwarte Spalten wie d_1, d_2, ...).")

    return wide_data


def compute_series_activity_table(wide_data: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt eine Tabelle mit einer Zeile pro Serie (aus Wide-Format):
    - series_id: entspricht der M5-Spalte 'id'
    - active_day_share: Anteil Tage mit Absatz > 0 über alle d_* Spalten
    - state_id, cat_id: aus den Metadaten
    """
    day_columns = [c for c in wide_data.columns if c.startswith("d_")]

    day_values = wide_data[day_columns].to_numpy()
    active_day_share = (day_values > 0).mean(axis=1)

    series_activity_table = pd.DataFrame({
        "series_id": wide_data["id"].astype(str),
        "active_day_share": active_day_share.astype(float),
        "state_id": wide_data["state_id"].astype(str),
        "cat_id": wide_data["cat_id"].astype(str),
    })

    return series_activity_table


def assign_demand_class(active_day_share: float) -> str:
    """Ordnet eine Serie einer Nachfrageklasse zu."""
    if active_day_share <= LOW_THRESHOLD:
        return "low"
    if active_day_share <= MID_THRESHOLD:
        return "mid"
    return "high"


def sample_stratified_by_state_and_category(series_table: pd.DataFrame, target_count: int):
    """Sampelt stratifiziert nach (state_id, cat_id) aus der gegebenen Serie-Tabelle."""
    if target_count <= 0 or len(series_table) == 0:
        return series_table.head(0).copy()

    selected_rows = []

    number_of_groups = series_table.groupby(["state_id", "cat_id"]).ngroups
    target_per_group = max(1, target_count // number_of_groups)

    for (state_id, category_id), group_dataframe in series_table.groupby(["state_id", "cat_id"]):
        if len(group_dataframe) < target_per_group:
            number_to_sample = len(group_dataframe)
        else:
            number_to_sample = target_per_group

        sampled_rows = group_dataframe.sample(
            n=number_to_sample,
            replace=False,
            random_state=RANDOM_SEED
        )

        selected_rows.append(sampled_rows)

    selected_table = pd.concat(selected_rows, ignore_index=True)

    # Falls zu wenig: zufällig auffüllen
    if len(selected_table) < target_count:
        remaining_table = series_table.loc[
            ~series_table["series_id"].isin(selected_table["series_id"])
        ]

        missing_count = target_count - len(selected_table)

        if missing_count > 0 and len(remaining_table) > 0:
            extra_rows = remaining_table.sample(
                n=min(missing_count, len(remaining_table)),
                replace=False,
                random_state=RANDOM_SEED
            )
            selected_table = pd.concat([selected_table, extra_rows], ignore_index=True)

    # Falls zu viele: zufällig kürzen
    if len(selected_table) > target_count:
        selected_table = selected_table.sample(
            n=target_count,
            replace=False,
            random_state=RANDOM_SEED
        )

    return selected_table


def save_subset(wide_data_subset: pd.DataFrame, output_path: Path) -> None:
    """Speichert das Subset als CSV (Wide-Format, gleiche Spalten wie Input)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".csv":
        raise ValueError("OUTPUT_NAME muss auf .csv enden (Wide-Export wie Input).")

    wide_data_subset.to_csv(output_path, index=False)


# ============================================================
# Hauptlogik
# ============================================================

def main() -> None:
    # 1) Daten laden (Wide)
    wide_data = load_wide_data(INPUT_PATH)

    # 2) Pro Serie Anteil aktiver Tage berechnen (series_level Tabelle)
    series_activity_table = compute_series_activity_table(wide_data)
    series_activity_table["demand_class"] = series_activity_table["active_day_share"].apply(assign_demand_class)

    # 3) Zielanzahl pro Nachfrageklasse bestimmen
    target_count_low = int(TARGET_NUMBER_OF_SERIES * DEMAND_CLASS_SHARES["low"])
    target_count_mid = int(TARGET_NUMBER_OF_SERIES * DEMAND_CLASS_SHARES["mid"])
    target_count_high = TARGET_NUMBER_OF_SERIES - target_count_low - target_count_mid

    # 4) Pro Nachfrageklasse stratifiziert nach (state_id, cat_id) sampeln
    low_series_table = series_activity_table[series_activity_table["demand_class"] == "low"]
    mid_series_table = series_activity_table[series_activity_table["demand_class"] == "mid"]
    high_series_table = series_activity_table[series_activity_table["demand_class"] == "high"]

    selected_low = sample_stratified_by_state_and_category(low_series_table, target_count_low)
    selected_mid = sample_stratified_by_state_and_category(mid_series_table, target_count_mid)
    selected_high = sample_stratified_by_state_and_category(high_series_table, target_count_high)

    selected_series_table = pd.concat([selected_low, selected_mid, selected_high], ignore_index=True)

    # 5) Wide-Daten auf die ausgewählten Serien filtern (EXAKT wie Input-Format)
    selected_series_ids = set(selected_series_table["series_id"].tolist())
    wide_data_subset = wide_data[wide_data["id"].astype(str).isin(selected_series_ids)].copy()

    # 6) Speichern (Wide Subset)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    save_subset(wide_data_subset, output_path)

    # 7) Summary
    print("Subset erstellt.")
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {output_path}")
    print(f"Anzahl Serien (soll): {TARGET_NUMBER_OF_SERIES}")
    print(f"Anzahl Serien (ist):  {wide_data_subset.shape[0]}")

    summary_by_class = selected_series_table["demand_class"].value_counts()
    print("\nVerteilung Nachfrageklassen (Series-Level):")
    print(summary_by_class)

    summary_by_state_cat = (
        selected_series_table
        .groupby(["demand_class", "state_id", "cat_id"])
        .size()
        .reset_index(name="n_series")
        .sort_values(["demand_class", "state_id", "cat_id"])
    )
    print("\nStichprobe je (Klasse, State, Cat) – erste 30 Zeilen:")
    print(summary_by_state_cat.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
