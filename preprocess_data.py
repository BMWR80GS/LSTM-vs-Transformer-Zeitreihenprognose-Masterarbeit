# ------------------------------------------------------------
# preprocess_data.py
#
# Ziel:
#   - M5 Rohdaten laden
#   - in Long-Format umformen
#   - Kalender + Preise joinen
#   - Datentypen vereinheitlichen / NaNs behandeln
#   - Train/Val/Test Splits pro Serie markieren
#   - y_log, price_s erzeugen
#   - y_z (train-only) + Lags/Rollings erzeugen
#   - Ergebnis als CSV + meta.json speichern
# ------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------

RAW_DIR = Path("data/raw")
SUBSET_DIR = Path("data/preprocessed/subsets")
OUT_DIR = Path("data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = SUBSET_DIR / "subset_200_series.csv"
CAL_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"

# Subset für Laptop-Experimente (später auf 2000/alle erhöhen)
MAX_SERIES = 200
SEED = 42

# Forecast-Horizont
HORIZON = 28

# Autoregressive Features wie im LSTM-Setup
LAGS = [1, 7, 28]
ROLL_WINDOWS = [7, 28]


# ------------------------------------------------------------
# 0) Kleine Hilfsfunktionen
# ------------------------------------------------------------

def ensure_files_exist():
    required = [SALES_FILE, CAL_FILE, PRICES_FILE]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Datei nicht gefunden: {p}\n"
                f"Erwartet wird:\n"
                f"  data/raw/calendar.csv\n"
                f"  data/raw/sell_prices.csv\n"
                f"  data/raw/sales_train_validation.csv\n"
            )


def log1p_float32(x: pd.Series) -> pd.Series:
    return np.log1p(x).astype(np.float32)


# ------------------------------------------------------------
# 1) Rohdaten laden
# ------------------------------------------------------------

# Sicherstellen, dass alle Dateien vorhanden sind
ensure_files_exist()

# Rohdaten laden
sales = pd.read_csv(SALES_FILE)
calendar = pd.read_csv(CAL_FILE)
prices = pd.read_csv(PRICES_FILE)


# ------------------------------------------------------------
# 2) Optionales Subsampling (auf Serienebene)
# ------------------------------------------------------------
# Hintergrund:
#   Für lokale Tests kann die Anzahl der Serien reduziert werden,
#   um Laufzeit und RAM-Bedarf zu senken.

if MAX_SERIES is not None and MAX_SERIES < len(sales):
    sales = sales.sample(n=MAX_SERIES, random_state=SEED).reset_index(drop=True)

# ------------------------------------------------------------
# 3) Sales: Wide -> Long
# ------------------------------------------------------------
# sales_train_validation.csv enthält die Tageswerte als Spalten d_1 ... d_1913.
# Für Modellierung und Join-Operationen ist Long-Format (eine Zeile pro Tag) einfacher.

# Alle ID Spalten
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Alle d_ Spalten, also die einzelnen Tage (d_1 ... d_1913)
d_cols = [c for c in sales.columns if c.startswith("d_")]

# In Long-Format umwandeln. Tage werden von Spalten in Zeilen transformiert.
# Spalte d enthält die Tageskennung (z.B. d_1), y die Verkaufszahl.
# Zeitreihenmodelle erwarten ein Long Format, also eine Zeile pro Serie + Tag.
df = sales.melt(
    id_vars=id_cols,
    value_vars=d_cols,
    var_name="d",
    value_name="y"
)


# ------------------------------------------------------------
# 4) Kalender-Join (calendar.csv)
# ------------------------------------------------------------
# Nur die wichtigen Spalten aus dem Kalender werden behalten.

cal_cols = [
    "d", "date", "wm_yr_wk",
    "wday", "month", "year",
    "event_name_1", "event_name_2",
    "snap_CA", "snap_TX", "snap_WI",
]

# Kleiner Kalender-Datensatz mit den definierten Spalten für Join
calendar_small = calendar[cal_cols].copy()

# Join auf d-Spalte (Tag)
df = df.merge(calendar_small, on="d", how="left")


# ------------------------------------------------------------
# 5) Preis-Join (sell_prices.csv)
# ------------------------------------------------------------
# Preise sind auf Woche (wm_yr_wk) + item_id + store_id definiert.
# Der Join der Preise erfolgt daher auf diese drei Spalten und so bekommt jeder Verkauf den passenden Preis.

df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

## Join Lücken der Preise analysieren
# Es gibt nicht für jede item_id + store_id + week einen Preis-Eintrag.Da manche Produkte erst ab einem bestimmten Datum verkauft wurden.
# # Die Verkaufszahlen sind aber für alle Tage definiert. Somit entstehen Lücken (NaNs) in der sell_price Spalte.
na_rate = df["sell_price"].isna().mean()
df_na = df[df["sell_price"].isna()]
na_count = df["sell_price"].isna().sum()
print("sell_price NaN rate:", na_rate, "NaN count:", na_count)
print("Beispielhafte fehlende Preise (erste 5 Zeilen):")
print(df_na.head(10))

# ------------------------------------------------------------
# 6) Datentypen und Basisbereinigung
# ------------------------------------------------------------

# Datum als datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# y: Falls im Datensatz NaNs (fehldene Werte) vorhanden sind, auf 0 setzen
df["y"] = df["y"].fillna(0.0).astype(np.float32)

# Anpassen des Datentyps auf float32. Speicherersparnis gegenüber float64.
df["sell_price"] = df["sell_price"].astype("float32")


# ------------------------------------------------------------
# 7) Serien-ID und Sortierung
# ------------------------------------------------------------

# Erstellung einer eindeutigen Serien-ID pro (store_id, item_id) Kombination.
# Jede Store Item Kombination ist eine eigene Zeitreihe.
df["series_id"] = (df["store_id"].astype(str) + "_" + df["item_id"].astype(str))

# Sortierung nach series_id
df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

# time_idx: fortlaufender Zeitindex pro Serie 1 bis 1912 (Anzahl der Tage)
df["time_idx"] = df.groupby("series_id").cumcount().astype(np.int32)


# ------------------------------------------------------------
# 8) Fehlende Preise füllen (Entscheidung teffen für NaNs)
# ------------------------------------------------------------
# Hintergrund:
#   Es besteht die Möglichkeit, dass Preise für bestimmte Tage in bestimmten Zeitreihen fehlen.
#   Um diese Lücken zu schließen, werden entweder Forward Fill und Backward Fill verwendet,
#   oder alle fehlenden Preise werden direkt mit 0.0 ersetzt.
#   Zusätzlich wird ein Flag gesetzt, das anzeigt, ob der Preis ursprünglich fehlte.

# Allgemeines price_missing Flag (1.0 wenn Preis ursprünglich fehlte, sonst 0.0)
df["price_missing"] = df["sell_price"].isna().astype(np.float32)

###### 1. Methode zum Füllen: Forward Fill und Backward Fill innerhalb jeder Serie
#ffill = forward fill
# forward fill bedeutet, dass fehlende Werte mit dem zuletzt bekannten Wert aufgefüllt werden.
#df["sell_price"] = df.groupby("series_id")["sell_price"].ffill()

# bfill = backward fill
# für den Beginn der Serie, falls dort noch NaNs sind und keine Preisdaten aus der Vergangenheit vorhanden sind:
# backward fill bedeutet, dass fehlende Werte mit dem nächstbekannten Wert aufgefüllt werden.
# Dies füllt verbleibende NaNs am Anfang der Serie.
#df["sell_price"] = df.groupby("series_id")["sell_price"].bfill()

# Falls immer noch NaNs vorhanden sind (z.B. wenn eine gesamte Serie keine Preisdaten hat), diese mit 0.0 ersetzen.
#df["sell_price"] = df["sell_price"].fillna(0.0).astype(np.float32)

###### 2. Methode zum Füllen: Alle Nans mit 0.0 ersetzen und zusätzliches price_missing Flag setzen

df["sell_price"] = df["sell_price"].fillna(0.0).astype(np.float32)

# ------------------------------------------------------------

# 9) SNAP und Event-Flags erzeugen
# ------------------------------------------------------------
# SNAP ist staatsspezifisch. Für eine einzige Feature-Spalte wird je nach state_id
# die passende SNAP-Spalte gewählt.

# SNAP wird zunächst auf 0.0 gesetzt (kein SNAP)
df["snap"] = 0.0

# Je nach state_id wird die passende SNAP-Spalte zugewiesen
df.loc[df["state_id"] == "CA", "snap"] = df.loc[df["state_id"] == "CA", "snap_CA"]
df.loc[df["state_id"] == "TX", "snap"] = df.loc[df["state_id"] == "TX", "snap_TX"]
df.loc[df["state_id"] == "WI", "snap"] = df.loc[df["state_id"] == "WI", "snap_WI"]

# SNAP NaNs (falls vorhanden) auf 0.0 setzen und in float32 umwandeln
df["snap"] = df["snap"].fillna(0.0).astype(np.float32)

# Events
# Wenn event_name_1 oder event_name_2 nicht NaN ist, wird das entsprechende Flag auf 1.0 gesetzt, sonst 0.0.    
df["has_event_1"] = df["event_name_1"].notna().astype(np.float32)
df["has_event_2"] = df["event_name_2"].notna().astype(np.float32)


# ------------------------------------------------------------
# 10) Splits train/val/test pro Serie markieren
# ------------------------------------------------------------
# Hintergrund:
#   Val = vorletzte 28 Tage, Test = letzte 28 Tage (je Serie).
#   Serien, die zu kurz sind, werden entfernt.

parts = []
kept_series = 0

for sid, g in df.groupby("series_id"):
    g = g.sort_values("time_idx").reset_index(drop=True)
    n = len(g)

    # Mindestlänge: train + val + test + Puffer
    if n < (3 * HORIZON + 20):
        continue

    test_start = n - HORIZON
    val_start = n - 2 * HORIZON

    g_train = g.iloc[:val_start].copy()
    g_val = g.iloc[val_start:test_start].copy()
    g_test = g.iloc[test_start:].copy()

    g_train["split"] = "train"
    g_val["split"] = "val"
    g_test["split"] = "test"

    parts.append(pd.concat([g_train, g_val, g_test], axis=0))
    kept_series += 1

df = pd.concat(parts, axis=0).reset_index(drop=True)


# ------------------------------------------------------------
# 11) Ziel-/Preis-Transformationen
# ------------------------------------------------------------
# Hintergrund:
#   - y_log: log1p(y) stabilisiert die Verteilung für Training
#   - price_s: log1p(price) betont relative Preisunterschiede

df["y_log"] = log1p_float32(df["y"])
df["price_s"] = log1p_float32(df["sell_price"])


# ------------------------------------------------------------
# 12) Kalenderfeatures skalieren
# ------------------------------------------------------------
# Hintergrund:
#   Kalenderfeatures werden auf [0,1] skaliert, um numerische Dominanz zu vermeiden.
#   Es handelt sich um einfache lineare Skalierung.

df["wday_s"] = (df["wday"].fillna(1).astype(np.float32) / 7.0).astype(np.float32)
df["month_s"] = (df["month"].fillna(1).astype(np.float32) / 12.0).astype(np.float32)

y_min = float(df["year"].min())
y_max = float(df["year"].max())
denom = max(1.0, y_max - y_min)
df["year_s"] = ((df["year"].fillna(y_min).astype(np.float32) - y_min) / denom).astype(np.float32)


# ------------------------------------------------------------
# 13) y_z (train-only) berechnen
# ------------------------------------------------------------
# Hintergrund:
#   Für y_z werden mu und sigma ausschließlich auf dem Train-Split bestimmt,
#   um Informationsleckage zu vermeiden.
#
# Hinweis:
#   Hier wird y_log standardisiert (nicht y). Das passt zum Training im Log-Raum
#   und ist numerisch stabil.

train_only = df[df["split"] == "train"].copy()

stats = (
    train_only.groupby("series_id")["y_log"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "mu", "std": "sigma"})
)

stats["sigma"] = stats["sigma"].replace(0.0, 1.0).fillna(1.0)

df = df.merge(stats, on="series_id", how="left")
df["mu"] = df["mu"].fillna(0.0).astype(np.float32)
df["sigma"] = df["sigma"].fillna(1.0).astype(np.float32)

df["y_z"] = ((df["y_log"] - df["mu"]) / df["sigma"]).astype(np.float32)


# ------------------------------------------------------------
# 14) Autoregressive Features aus y_z (Lags / Rollings)
# ------------------------------------------------------------
# Hintergrund:
#   Diese Features wurden zuvor im LSTM-Training erzeugt.
#   Für einheitliche Datengrundlage werden sie hier bereits im Preprocessing berechnet.

df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

# Lags
for lag in LAGS:
    df[f"y_z_lag_{lag}"] = df.groupby("series_id")["y_z"].shift(lag)

# Rolling (immer nur Vergangenheit -> shift(1))
for w in ROLL_WINDOWS:
    shifted = df.groupby("series_id")["y_z"].shift(1)

    df[f"y_z_roll_mean_{w}"] = (
        shifted.groupby(df["series_id"])
        .rolling(window=w, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"y_z_roll_std_{w}"] = (
        shifted.groupby(df["series_id"])
        .rolling(window=w, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

# NaNs am Serienanfang neutral auffüllen (0 im standardisierten Raum)
lag_cols = [f"y_z_lag_{l}" for l in LAGS]
roll_cols = [f"y_z_roll_mean_{w}" for w in ROLL_WINDOWS] + [f"y_z_roll_std_{w}" for w in ROLL_WINDOWS]
df[lag_cols + roll_cols] = df[lag_cols + roll_cols].fillna(0.0).astype(np.float32)


# ------------------------------------------------------------
# 15) Spaltenauswahl
# ------------------------------------------------------------

keep_cols = [
    "series_id", "time_idx", "date", "split",
    "store_id", "item_id", "dept_id", "cat_id", "state_id",
    "y", "y_log", "mu", "sigma", "y_z",
    "sell_price", "price_s", "price_missing", "snap","wday",
    "wday_s", "month", "month_s", "year_s",
    "has_event_1", "has_event_2",
]

keep_cols += lag_cols
keep_cols += [f"y_z_roll_mean_{w}" for w in ROLL_WINDOWS]
keep_cols += [f"y_z_roll_std_{w}" for w in ROLL_WINDOWS]

df = df[keep_cols].copy()


# ------------------------------------------------------------
# 16) Speichern + Meta
# ------------------------------------------------------------

out_csv = OUT_DIR / "m5_long.csv"
out_meta = OUT_DIR / "meta.json"

df.to_csv(out_csv, index=False)

meta = {
    "seed": SEED,
    "max_series_requested": MAX_SERIES,
    "series_kept_after_length_filter": kept_series,
    "horizon": HORIZON,
    "lags": LAGS,
    "rolling_windows": ROLL_WINDOWS,
    "notes": (
        "Step-by-step preprocessing: melt -> calendar join -> price join -> "
        "types/NaNs -> per-series splits -> transforms (y_log, price_s) -> "
        "calendar scaling -> train-only y_z -> y_z lags/rollings -> save."
    )
}

with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved:", str(out_csv))
print("Meta :", str(out_meta))
print("Rows:", len(df), "Series:", df["series_id"].nunique())
print("Date range:", str(df["date"].min()), "->", str(df["date"].max()))
print("Split counts:\n", df["split"].value_counts())
