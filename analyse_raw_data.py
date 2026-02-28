# -----------------------------------------------------------------------------
# Reads the original M5 sales file (sales_train_validation.csv, wide format)
# and produces an initial EDA + German-labeled plots for a Master's thesis.
#
# Adds structure plots: number of states, stores, categories, departments, items
# and distributions (e.g., series per store / per department / per category).

#
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy import stats

### Constants
# Wähle das Datenset für die Analyse (Full; Sub_Small; Sub_Big)
Dataset = "Sub_Small"  # "Full", "Sub_Small", "Sub_Big"
if Dataset == "Full":
    SALES_FILE = Path("data/raw/sales_train_evaluation.csv")
    NUMBER_OF_SERIES = 30490
elif Dataset == "Sub_Small":
    SALES_FILE = Path("data/preprocessed/subsets/subset_200_series.csv")
    NUMBER_OF_SERIES = 200
elif Dataset == "Sub_Big":
    SALES_FILE = Path("data/preprocessed/subsets/subset_2000_series.csv")
    NUMBER_OF_SERIES = 2000
else:
    raise ValueError(f"Unknown Dataset selection: {Dataset}")

CAL_FILE = Path("data/raw/calendar.csv")
OUT_BASE = Path("data/Analyse Dataset") / Dataset


def create_eda_run_folder():
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = Dataset + "__" + ts
    out = OUT_BASE / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_sales_and_calendar():
    sales = pd.read_csv(SALES_FILE)

    cal = None
    if CAL_FILE.exists():
        cal = pd.read_csv(CAL_FILE, usecols=["d", "date"])
        cal["date"] = pd.to_datetime(cal["date"])
    else:
        print("WARN: calendar.csv not found -> plots will use d_* index axis (not calendar dates).")

    return sales, cal

def get_day_cols(sales: pd.DataFrame) -> list[str]:
    return [i for i in sales.columns if i.startswith("d_")]


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------- Auswertungen --------------------------------
# Umwandlung von Wide to Long
def wide_to_daily_total(sales: pd.DataFrame, d_cols: list[str], cal: pd.DataFrame):
    total = sales[d_cols].sum(axis=0).to_frame("total_sales").reset_index().rename(columns={"index": "d"})
    total = total.merge(cal, on="d", how="left")
    total = total.sort_values("date").reset_index(drop=True)
    return total


def compute_basic_stats(sales: pd.DataFrame, d_cols: list[str]):
    # Numpy Array der Verkaufszahlen erstellen
    y = sales[d_cols].to_numpy(dtype=np.int32)

    series_sum = y.sum(axis=1)
    series_mean = y.mean(axis=1)
    series_zero_share = (y == 0).mean(axis=1)

    stats = {
        "Anzahl Serien": int(len(sales)),
        "Anzahl Tage": int(len(d_cols)),
        "Summe des Absatzes über alle Tage (alle Serien)": float(series_sum.sum()),
        "Quantil p50 Summe aller Verkäufe (alle Serien)": float(np.quantile(series_sum, 0.5)),
        "Quantil p90 Summe aller Verkäufe (alle Serien)": float(np.quantile(series_sum, 0.9)),
        "Quantil p99 Summe aller Verkäufe (alle Serien)": float(np.quantile(series_sum, 0.99)),
        "Durchschnittlicher Absatz über alle Tage (alle Serien)": float(series_sum.mean()),
        "Durchschnittlicher Absatz pro Tag (alle Serien)": float(y.mean()),
        "Quantil p50 Absatz pro Tag (alle Serien)": int(np.quantile(y, 0.5)),
        "Quantil p75 Absatz pro Tag (alle Serien)": int(np.quantile(y, 0.75)),
        "Quantil p90 Absatz pro Tag (alle Serien)": int(np.quantile(y, 0.9)),
        "Quantil p99 Absatz pro Tag (alle Serien)": int(np.quantile(y, 0.99)),
        "Median Absatz über alle Tage (alle Serien)": float(np.median(series_sum)),
        "Durchschnittlicher Absatz pro Tag (alle Serien)": float(series_mean.mean()),
        "Durschnittlicher Anteil von 0 Absatz Tagen (alle Serien)": float(series_zero_share.mean()),
        "Median Anteil von 0 Absatz Tagen (alle Serien)": float(np.median(series_zero_share)),
    }
    # Quantile für Absatz pro Tag berechnen
    quantile = np.linspace(0.0, 1.0, 200)
    quantile_values = np.quantile(y, quantile)

    # Erstes Quantil mit positivem Absatz
    positive_sales = quantile_values > 0
    first_positive_index = np.argmax(positive_sales)
    first_positive_quantile = quantile[first_positive_index]
   
    # Schlüsselquantile für das Plotten markieren
    key_quantile = [0.5, 0.75, 0.9, 0.99, first_positive_quantile]
    key_quantile_values = np.quantile(y, key_quantile)
    return stats, quantile, quantile_values, key_quantile, key_quantile_values


def compute_structure_counts(sales: pd.DataFrame):
    def number_nunique(col: str) -> int | None:
        return int(sales[col].nunique()) if col in sales.columns else None

    counts = {
        "n_states": number_nunique("state_id"),
        "n_stores": number_nunique("store_id"),
        "n_categories": number_nunique("cat_id"),
        "n_departments": number_nunique("dept_id"),
        "n_items": number_nunique("item_id"),
    }
    return counts


# ---------------------------- plotting (German labels) ------------------------
def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_total_sales_over_time(daily_total: pd.DataFrame, out_path: Path):
    plt.figure()
    if "date" in daily_total.columns and daily_total["date"].notna().any():
        plt.plot(daily_total["date"], daily_total["total_sales"])
        plt.xlabel("Datum")
    else:
        plt.plot(np.arange(len(daily_total)), daily_total["total_sales"])
        plt.xlabel("Tag-Index (d_*)")
    plt.ylabel("Gesamtabsatz (Summe über alle Zeitreihen)")
    plt.title("M5: Gesamtabsatz über die Zeit (Subset)")
    plt.grid(True)
    _savefig(out_path)


def plot_quantile_sales(quantile: list, quantile_values: list, key_quantile: list, key_quantile_values: list, out_path: Path):
    # Erstes Quantil mit positivem Absatz
    positive_sales = quantile_values > 0
    first_positive_index = np.argmax(positive_sales)
    first_positive_quantile = quantile[first_positive_index]
    first_positive_value = quantile_values[first_positive_index]

    # max Wert für y-Achse und min Wert für x-Achse bestimmen
    max_y = max(quantile_values)
    min_x = min(quantile)

    # Beschriftung der Punkte anpassen
    offsets = {
        0.50: (-0.20,  max_y*0.1),  
        0.75: (-0.20,  max_y*0.1),
        0.90: (-0.15,  max_y*0.1),
        0.99: (-0.18, max_y*0.5),
        float(first_positive_quantile): (min_x, max_y),   
    }

    plt.plot(quantile, quantile_values)

    # Wichtige Quantile hervorheben
    plt.scatter(key_quantile, key_quantile_values, color="red", zorder=3)

    # Erstes Positiv-Quantil hervorheben
    plt.scatter([first_positive_quantile], [key_quantile_values[-1]], color="green", zorder=4)

    # Beschriftungen an den Punkten
    for quantil, value in zip(key_quantile, key_quantile_values):
        xoffset, yoffset = offsets[float(quantil)]
        if(quantil != first_positive_quantile):
            plt.annotate(f"Quantil: {quantil:.2f} \nAbsatz: {value:.0f}",
                    xy=(quantil, value),
                    # Textposition anpassen
                    xytext=(quantil+xoffset, yoffset),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=10)

    plt.xlabel("Quantil")
    plt.ylabel("Tägliche Absatzzahlen")
    plt.title(f"Tägliche Absatzzahlen nach Quantilen ({NUMBER_OF_SERIES} Zeitreihen)")
    plt.grid(True)

    # Textbox für erstes Positiv-Quantil
    ax = plt.gca()
    ax.text(
        0.02, 0.98,  # x,y in Achsen-Koordinaten
        f"Erstes Positiv-Quantil: {first_positive_quantile:.2f}\nAbsatz: {first_positive_value:.0f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="green",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.9),
        zorder=10
    )
    _savefig(out_path)


def plot_total_sales_histogram(sales: pd.DataFrame, d_cols: list[str], out_path: Path):
    totals = sales[d_cols].sum(axis=1).to_numpy(dtype=np.float32)
    plt.figure()
    plt.hist(totals, bins=60)
    plt.xlabel("Gesamtabsatz je Zeitreihe (Summe über alle Tage)")
    plt.ylabel("Anzahl Zeitreihen")
    plt.title("Verteilung: Gesamtabsatz pro Zeitreihe")
    plt.grid(True)
    _savefig(out_path)


def plot_zero_share_histogram(sales: pd.DataFrame, d_cols: list[str], out_path: Path):
    y = sales[d_cols].to_numpy(dtype=np.float32)
    zero_share = (y == 0).mean(axis=1)*100  # in %
    plt.figure()
    plt.hist(zero_share, bins=50)
    plt.xlabel("%-Anteil Null-Tage je Zeitreihe")
    plt.ylabel("Anzahl Zeitreihen")
    plt.title("Sparsity: Anteil an Tagen mit 0 Absatz je Zeitreihe")
    plt.grid(True)
    _savefig(out_path)


def plot_top_series_bars(sales: pd.DataFrame, d_cols: list[str], out_path: Path, k: int = 20):
    totals = sales[d_cols].sum(axis=1)
    top = totals.sort_values(ascending=False).head(k)
    labels = sales.loc[top.index, "id"].astype(str).tolist() if "id" in sales.columns else [str(i) for i in top.index]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top))[::-1], top.values[::-1])
    plt.yticks(range(len(top))[::-1], labels[::-1], fontsize=8)
    plt.xlabel("Gesamtabsatz")
    plt.title(f"Top-{k} Zeitreihen nach Gesamtabsatz")
    plt.grid(True, axis="x")
    _savefig(out_path)


def plot_state_totals_over_time(
    sales: pd.DataFrame, d_cols: list[str], cal: pd.DataFrame, out_path: Path):
    if "state_id" not in sales.columns:
        return

    by_state = sales.groupby("state_id")[d_cols].sum()

    if cal is not None and cal["date"].notna().any():
        x = cal.set_index("d").loc[d_cols, "date"].values
        xlabel = "Datum"
    else:
        x = np.arange(len(d_cols))
        xlabel = "Tag-Index (d_*)"

    plt.figure(figsize=(10, 5))
    for state in by_state.index:
        plt.plot(x, by_state.loc[state].values, label=str(state))
    plt.xlabel(xlabel)
    plt.ylabel("Gesamtabsatz")
    plt.title("Gesamtabsatz über die Zeit nach Bundesstaat")
    plt.grid(True)
    plt.legend()
    _savefig(out_path)


def plot_weekly_seasonality_avg(sales: pd.DataFrame, cal: pd.DataFrame | None, d_cols: list[str], out_path: Path):
    if cal is None or not {"d", "date"}.issubset(cal.columns):
        return

    totals = sales[d_cols].sum(axis=0).to_frame("total_sales").reset_index().rename(columns={"index": "d"})
    totals = totals.merge(cal, on="d", how="left").dropna(subset=["date"])
    totals["weekday"] = totals["date"].dt.day_name()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = totals.groupby("weekday")["total_sales"].mean().reindex(order)

    # translate weekday labels to German
    german = {
        "Monday": "Montag",
        "Tuesday": "Dienstag",
        "Wednesday": "Mittwoch",
        "Thursday": "Donnerstag",
        "Friday": "Freitag",
        "Saturday": "Samstag",
        "Sunday": "Sonntag",
    }
    labels = [german.get(x, x) for x in g.index]

    plt.figure()
    plt.bar(np.arange(len(g)), g.values)
    plt.xticks(np.arange(len(g)), labels, rotation=30, ha="right")
    plt.ylabel("Durchschnittlicher Gesamtabsatz")
    plt.title("Wochensaisonalität: Durchschnittlicher Gesamtabsatz nach Wochentag")
    plt.grid(True, axis="y")
    _savefig(out_path)


# -------------------------  structure plots -------------------------------
def plot_structure_overview_bar(struct_counts: dict, out_path: Path):
    # filter None values if columns are missing
    keys = [k for k, v in struct_counts.items() if v is not None]
    vals = [struct_counts[k] for k in keys]

    # nicer x labels in German
    label_map = {
        "n_states": "Bundesstaaten",
        "n_stores": "Stores",
        "n_categories": "Kategorien",
        "n_departments": "Departments",
        "n_items": "Items",
    }
    xlabels = [label_map.get(k, k) for k in keys]

    plt.figure()
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), xlabels, rotation=25, ha="right")
    plt.ylabel("Anzahl")
    plt.title("Strukturübersicht: Anzahl von Hierarchie-Entitäten (im Subset)")
    plt.grid(True, axis="y")
    _savefig(out_path)


def plot_distribution_count_by_group(sales: pd.DataFrame, group_col: str, out_path: Path, top_k: int = 25):
    if group_col not in sales.columns:
        return
    counts = sales[group_col].value_counts().head(top_k).sort_values(ascending=True)

    title_map = {
        "store_id": "Zeitreihen je Store (Top)",
        "dept_id": "Zeitreihen je Department (Top)",
        "cat_id": "Zeitreihen je Kategorie (Top)",
        "state_id": "Zeitreihen je Bundesstaat",
    }
    xlabel_map = {
        "store_id": "Store",
        "dept_id": "Department",
        "cat_id": "Kategorie",
        "state_id": "Bundesstaat",
    }

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(counts)), counts.values)
    plt.yticks(range(len(counts)), counts.index.astype(str).tolist(), fontsize=8)
    plt.xlabel("Anzahl Zeitreihen")
    plt.ylabel(xlabel_map.get(group_col, group_col))
    plt.title(title_map.get(group_col, f"Zeitreihen je {group_col} (Top)"))
    plt.grid(True, axis="x")
    _savefig(out_path)


def plot_items_per_category(sales: pd.DataFrame, out_path: Path):
    # unique items per category
    if "cat_id" not in sales.columns or "item_id" not in sales.columns:
        return

    tmp = sales.groupby("cat_id")["item_id"].nunique().sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(range(len(tmp)), tmp.values)
    plt.yticks(range(len(tmp)), tmp.index.astype(str).tolist())
    plt.xlabel("Anzahl unterschiedlicher Items")
    plt.ylabel("Kategorie")
    plt.title("Anzahl unterschiedlicher Items je Kategorie")
    plt.grid(True, axis="x")
    _savefig(out_path)


def plot_departments_per_store(sales: pd.DataFrame, out_path: Path):
    if "store_id" not in sales.columns or "dept_id" not in sales.columns:
        return
    tmp = sales.groupby("store_id")["dept_id"].nunique().sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(tmp)), tmp.values)
    plt.yticks(range(len(tmp)), tmp.index.astype(str).tolist(), fontsize=8)
    plt.xlabel("Anzahl Departments")
    plt.ylabel("Store")
    plt.title("Anzahl Departments je Store")
    plt.grid(True, axis="x")
    _savefig(out_path)


# ---------------------------- exports -----------------------------------------
def export_series_stats_table(sales: pd.DataFrame, d_cols: list[str], out_path: Path):
    y = sales[d_cols].to_numpy(dtype=np.float32)
    totals = y.sum(axis=1)
    mean = y.mean(axis=1)
    zero_share = (y == 0).mean(axis=1)

    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    keep = [c for c in cols if c in sales.columns]

    out = sales[keep].copy()
    out["series_total_sales"] = totals
    out["series_daily_mean"] = mean
    out["series_zero_share"] = zero_share

    out.to_csv(out_path, index=False)


# ---------------------------- main --------------------------------------------
def main():
    # Ordner erstellen
    run_dir = create_eda_run_folder()

    # Daten laden
    sales, cal = load_sales_and_calendar()

    # Tags-Spalten identifizieren
    d_cols = get_day_cols(sales)

    # Auswertungen starten
    basic_stats, quantiles, quantile_values, key_quantile, key_quantile_values = compute_basic_stats(sales, d_cols)
    struct_counts = compute_structure_counts(sales)

    config = {
        "source_sales_file": str(SALES_FILE),
        "source_calendar_file": str(CAL_FILE),
        "n_series_in_raw": int(len(sales)),
        "n_series_in_subset": int(len(sales)),
        "n_days": int(len(d_cols)),
    }

    # Speichern der Ergebnisse als JSON
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "summary_basic_stats.json", basic_stats)
    save_json(run_dir / "summary_structure_counts.json", struct_counts)

    # Erstellen und Speichern der Ergebnisse als CSV
    daily_total = wide_to_daily_total(sales, d_cols, cal)
    daily_total.to_csv(run_dir / "daily_total_sales.csv", index=False)
    export_series_stats_table(sales, d_cols, run_dir / "series_stats.csv")

    # Abbildungen erstellen und speichern
    plot_quantile_sales(quantiles, quantile_values, key_quantile, key_quantile_values, run_dir / "00_quantile_absatz_pro_tag.png")
    plot_total_sales_over_time(daily_total, run_dir / "01_gesamtabsatz_ueber_zeit.png")
    plot_total_sales_histogram(sales, d_cols, run_dir / "02_verteilung_gesamtabsatz_je_zeitreihe.png")
    #plot_total_sales_normal_distribution(sales, d_cols, run_dir / "02b_normalverteilung_gesamtabsatz_je_zeitreihe.png")
    plot_zero_share_histogram(sales, d_cols, run_dir / "03_verteilung_nullanteil_je_zeitreihe.png")
    plot_top_series_bars(sales, d_cols, run_dir / "04_top20_zeitreihen_nach_gesamtabsatz.png", k=20)
    plot_state_totals_over_time(sales, d_cols, cal, run_dir / "05_gesamtabsatz_nach_bundesstaat_ueber_zeit.png")
    plot_weekly_seasonality_avg(sales, cal, d_cols, run_dir / "06_wochensaisonalitaet_nach_wochentag.png")
    plot_structure_overview_bar(struct_counts, run_dir / "07_strukturuebersicht_anzahlen.png")
    plot_distribution_count_by_group(sales, "state_id", run_dir / "08_zeitreihen_je_bundesstaat.png", top_k=10)
    plot_distribution_count_by_group(sales, "store_id", run_dir / "09_zeitreihen_je_store_top25.png", top_k=25)
    plot_distribution_count_by_group(sales, "cat_id", run_dir / "10_zeitreihen_je_kategorie.png", top_k=20)
    plot_distribution_count_by_group(sales, "dept_id", run_dir / "11_zeitreihen_je_department_top25.png", top_k=25)
    plot_items_per_category(sales, run_dir / "12_items_je_kategorie.png")
    plot_departments_per_store(sales, run_dir / "13_departments_je_store.png")

    print("\nAnalysing finished. Outputs saved to:")
    print(run_dir.resolve())

    print("\nKey basic stats:")
    for k, v in basic_stats.items():
        print(f"  {k}: {v}")

    print("\nStructure counts:")
    for k, v in struct_counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()