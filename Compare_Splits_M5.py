# Analyse des Datensatzes auf den Vergleich der Komplexität von den Splits 'train', 'val' und 'test'
# Hintergrund:  
#   Beweis dafür, dass die 'val' und  'test' Periode komplexer ist als die 'train' Periode.
#   Damit können die Ergebnisse der Arbeit besser eingeordnet werden.
# Vorgehen:
#   - Berechnung der MSE und MAE pro Serie und Split
#   - Für Naive Baseline (Repeat-1) und Seasonal Naive Baseline (Repeat-7)


import time
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_DIR = Path("data") / "preprocessed"
CSV_PATH = DATA_DIR / "m5_long.csv"
PARQUET_PATH = DATA_DIR / "m5_long.parquet"

HORIZON = 28
SEASONALITY_1 = 1
SEASONALITY_7 = 7
SPLIT_NAME = "test"   # "val" or "test"
RUNS_DIR = Path("runs")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError("m5_long.csv")
    return df


# # ---------------------------------------------------------------------
# # MASE denominator (in-sample seasonal naive error on TRAIN)
# # ---------------------------------------------------------------------
# def compute_mase_denominators(df: pd.DataFrame, seasonality: int = 7) -> dict:
#     """Compute per-series MASE denominators on TRAIN split.

#     den_i = mean(|y_t - y_{t-seasonality}|) over TRAIN observations.

#     If a series has fewer than (seasonality + 1) TRAIN points, or the
#     denominator is 0, we store NaN and skip it during MASE aggregation.
#     """
#     denoms: dict[str, float] = {}

#     for sid, g in df.groupby("series_id"):
#         g = g.sort_values("time_idx")
#         g_train = g[g["split"] == "train"]

#         y_tr = g_train["y"].astype(np.float32).values
#         if len(y_tr) <= seasonality:
#             denoms[sid] = np.nan
#             continue

#         diffs = np.abs(y_tr[seasonality:] - y_tr[:-seasonality])
#         den = float(np.mean(diffs)) if len(diffs) else np.nan

#         # Avoid division-by-zero explosions; follow common practice: mark as NaN
#         # Wenn der Nenner 0 oder nicht endlich ist, wird MASE für diese Serie nicht berechnet.
#         if not np.isfinite(den) or den <= 0.0:
#             denoms[sid] = np.nan
#         else:
#             denoms[sid] = den

#     return denoms


# ---------------------------------------------------------------------
# Naive Baseline (Repeat-1)
# ---------------------------------------------------------------------
def naive_baseline(df: pd.DataFrame, split_name: str):
    maes = []
    mses = []
    #mases = []
    n_windows = 0
    #n_windows_mase = 0

    for sid, g in df.groupby("series_id"):
        g = g.sort_values("time_idx").reset_index(drop=True)

        y = g["y"].astype(np.float32).values
        splits = g["split"].values
        #den = denoms.get(sid, np.nan)

        for t in range(SEASONALITY_1, len(g) - HORIZON + 1):
            if splits[t] != split_name:
                continue

            # True values for the forecast horizon
            y_true = y[t : t + HORIZON]
            # Forecast values: repeat values from 7 days earlier
            y_pred = y[t - SEASONALITY_1 : t - SEASONALITY_1 + HORIZON]

            if len(y_pred) != HORIZON:
                continue

            mae = float(np.mean(np.abs(y_true - y_pred)))
            mse = float(np.mean((y_true - y_pred) ** 2))

            maes.append(mae)
            mses.append(mse)
            n_windows += 1

            # MASE: mae (forecast error) / den (training in-sample seasonal naive error)
            # if np.isfinite(den) and den > 0.0:
            #     mases.append(mae / den)
            #     n_windows_mase += 1

    return {
        'Baseline': 'Seasonal Naive (Repeat-1)',
        "split": split_name,
        "series": df["series_id"].nunique(),
        "n_windows": n_windows,
        #"n_windows_mase": n_windows_mase,
        "mae": float(np.mean(maes)) if maes else np.nan,
        "mse": float(np.mean(mses)) if mses else np.nan,
        #"mase": float(np.mean(mases)) if mases else np.nan,
    }


# ---------------------------------------------------------------------
# Seasonal Naive Baseline (Repeat-7)
# ---------------------------------------------------------------------
def seasonal_naive_baseline(df: pd.DataFrame, split_name: str):
    maes = []
    mses = []
    #mases = []
    n_windows = 0
    #n_windows_mase = 0

    for sid, g in df.groupby("series_id"):
        g = g.sort_values("time_idx").reset_index(drop=True)

        y = g["y"].astype(np.float32).values
        splits = g["split"].values
        #den = denoms.get(sid, np.nan)

        for t in range(SEASONALITY_7, len(g) - HORIZON + 1):
            if splits[t] != split_name:
                continue

            # True values for the forecast horizon
            y_true = y[t : t + HORIZON]
            # Forecast values: repeat values from 7 days earlier
            y_pred = y[t - SEASONALITY_7 : t - SEASONALITY_7 + HORIZON]

            if len(y_pred) != HORIZON:
                continue

            mae = float(np.mean(np.abs(y_true - y_pred)))
            mse = float(np.mean((y_true - y_pred) ** 2))

            maes.append(mae)
            mses.append(mse)
            n_windows += 1

            # MASE: mae (forecast error) / den (training in-sample seasonal naive error)
            # if np.isfinite(den) and den > 0.0:
            #     mases.append(mae / den)
            #     n_windows_mase += 1

    return {
        'Baseline': 'Seasonal Naive (Repeat-7)',
        "split": split_name,
        "series": df["series_id"].nunique(),
        "n_windows": n_windows,
        #"n_windows_mase": n_windows_mase,
        "mae": float(np.mean(maes)) if maes else np.nan,
        "mse": float(np.mean(mses)) if mses else np.nan,
        #"mase": float(np.mean(mases)) if mases else np.nan,
    }

# def seasonal_naive_by_horizon(df: pd.DataFrame, split_name: str, denoms: dict) -> pd.DataFrame:
#     mae_h = [[] for _ in range(HORIZON)]
#     mse_h = [[] for _ in range(HORIZON)]
#     mase_h = [[] for _ in range(HORIZON)]

#     for sid, g in df.groupby("series_id"):
#         g = g.sort_values("time_idx").reset_index(drop=True)

#         y = g["y"].astype(np.float32).values
#         splits = g["split"].values
#         den = denoms.get(sid, np.nan)

#         for t in range(SEASONALITY, len(g) - HORIZON + 1):
#             if splits[t] != split_name:
#                 continue

#             y_true = y[t : t + HORIZON]
#             y_pred = y[t - SEASONALITY : t - SEASONALITY + HORIZON]

#             if len(y_pred) != HORIZON:
#                 continue

#             err = y_true - y_pred
#             ae = np.abs(err)
#             se = err ** 2

#             for h in range(HORIZON):
#                 mae_h[h].append(float(ae[h]))
#                 mse_h[h].append(float(se[h]))
#                 if np.isfinite(den) and den > 0.0:
#                     mase_h[h].append(float(ae[h] / den))

#     rows = []
#     for h in range(HORIZON):
#         rows.append(
#             {
#                 "horizon": h + 1,
#                 "mae": float(np.mean(mae_h[h])) if mae_h[h] else np.nan,
#                 "mse": float(np.mean(mse_h[h])) if mse_h[h] else np.nan,
#                 "mase": float(np.mean(mase_h[h])) if mase_h[h] else np.nan,
#                 "n": int(len(mae_h[h])),
#                 "n_mase": int(len(mase_h[h])),
#             }
#         )
#     return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Logging / Exports (analog zur Trainingsskript-Struktur)
# ---------------------------------------------------------------------
def get_system_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def create_run_folder(base_dir: Path):
    run_dir = base_dir / "Compare_Splits_M5"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# def save_plots(run_dir: Path, by_horizon_df: pd.DataFrame, results: dict) -> tuple[Path, Path]:
#     loss_path = run_dir / "loss.png"
#     metrics_path = run_dir / "metrics.png"

#     # loss.png: MAE + MASE per horizon (two lines)
#     plt.figure()
#     plt.plot(by_horizon_df["horizon"], by_horizon_df["mae"], marker="o", label="MAE")
#     if "mase" in by_horizon_df.columns:
#         plt.plot(by_horizon_df["horizon"], by_horizon_df["mase"], marker="o", label="MASE")
#     plt.title("Seasonal Naive: Fehler pro Horizont")
#     plt.xlabel("Horizont (Tage)")
#     plt.ylabel("Wert")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(loss_path, dpi=150)
#     plt.close()

#     # metrics.png: bar plot MAE/MSE/MASE (overall)
#     plt.figure()
#     labels = ["MAE", "MSE", "MASE"]
#     vals = [float(results["mae"]), float(results["mse"]), float(results["mase"])]
#     plt.bar(labels, vals)
#     plt.title("Baseline-Metriken (Seasonal Naive)")
#     plt.ylabel("Wert")
#     plt.grid(True, axis="y")
#     plt.tight_layout()
#     plt.savefig(metrics_path, dpi=150)
#     plt.close()

#     return loss_path, metrics_path


def save_forecast_example(df: pd.DataFrame, split_name: str, out_path: Path, baseline: str) -> bool:
    rng = np.random.default_rng(42)

    SEASONALITY = SEASONALITY_7 if baseline == "Seasonal Naive" else SEASONALITY_1

    candidates = []
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("time_idx").reset_index(drop=True)
        splits = g["split"].values
        for t in range(SEASONALITY, len(g) - HORIZON + 1):
            if splits[t] == split_name:
                candidates.append((sid, t))
                break

    if not candidates:
        return False

    sid, t = candidates[int(rng.integers(5, len(candidates)))]
    g = df[df["series_id"] == sid].sort_values("time_idx").reset_index(drop=True)
    y = g["y"].astype(np.float32).values

    y_true = y[t : t + HORIZON]
    y_pred = y[t - SEASONALITY : t - SEASONALITY + HORIZON]

    plt.figure()
    plt.plot(y_true, label="True")
    if( baseline == "Naive"):
        plt.plot(y_pred, label="Pred (repeat-1)", linestyle="--")
    else:
        plt.plot(y_pred, label="Pred (repeat-7)", linestyle="--")
    plt.title(f"Beispielforecast (Split={split_name})\n{baseline}\nSeries: {sid}")
    plt.xlabel("Forecast (in Tagen)")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(SPLIT_NAME: str):
    total_start = time.time()

    print("Loading data...")

    df = load_data()

    config = {
        "model": "SeasonalNaiveRepeat7",
        "data_dir": str(DATA_DIR),
        "split": SPLIT_NAME,
        "series": df["series_id"].nunique(),
        "horizon": HORIZON,
        "csv_path": str(CSV_PATH),
        "parquet_path": str(PARQUET_PATH),
        "mase_denominator": "mean(|y_t - y_{t-7}|) on TRAIN split per series",
    }

    run_dir = create_run_folder(RUNS_DIR)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "system_info.json", get_system_info())

    #denoms = compute_mase_denominators(df, seasonality=SEASONALITY)

    print(f"Evaluating Naive Baseline on split='{SPLIT_NAME}'")
    results_naive = naive_baseline(df, SPLIT_NAME)

    print("\n=== Naive Baseline (Repeat-1) Results ===")
    print(f"Split         : {results_naive['split']}")
    print(f"Series        : {df["series_id"].nunique()}")
    print(f"Windows       : {results_naive['n_windows']}")
    #print(f"Windows (MASE): {results_seasonal_naive['n_windows_mase']}")
    print(f"MAE           : {results_naive['mae']:.4f}")
    print(f"MSE           : {results_naive['mse']:.4f}")
    #print(f"MASE          : {results_seasonal_naive['mase']:.4f}")
    print("=====================================")

    print(f"Evaluating Seasonal Naive Baseline on split='{SPLIT_NAME}'")
    results_seasonal_naive = seasonal_naive_baseline(df, SPLIT_NAME)
    #by_horizon_df = seasonal_naive_by_horizon(df, SPLIT_NAME)

    print("\n=== Seasonal Naive Baseline (Repeat-7) Results ===")
    print(f"Split         : {results_seasonal_naive['split']}")
    print(f"Series        : {df["series_id"].nunique()}")
    print(f"Windows       : {results_seasonal_naive['n_windows']}")
    #print(f"Windows (MASE): {results_seasonal_naive['n_windows_mase']}")
    print(f"MAE           : {results_seasonal_naive['mae']:.4f}")
    print(f"MSE           : {results_seasonal_naive['mse']:.4f}")
    #print(f"MASE          : {results_seasonal_naive['mase']:.4f}")
    print("=====================================")

    summary_naive = {
        "split": results_naive["split"],
        "n_windows": results_naive["n_windows"],
        #"n_windows_mase": results_naive["n_windows_mase"],
        "mae": results_naive["mae"],
        "mse": results_naive["mse"],
        #"mase": results_naive["mase"],
        #"total_time_sec": float(time.time() - total_start),
        #"run_dir": str(run_dir),
    }

    save_json(run_dir / "summary_naive.json", summary_naive)

    summary_seasonal_naive = {
        "split": results_seasonal_naive["split"],
        "n_windows": results_seasonal_naive["n_windows"],
        #"n_windows_mase": results_seasonal_naive["n_windows_mase"],
        "mae": results_seasonal_naive["mae"],
        "mse": results_seasonal_naive["mse"],
        #"mase": results_seasonal_naive["mase"],
        #"total_time_sec": float(time.time() - total_start),
        #"run_dir": str(run_dir),
    }

    save_json(run_dir / "summary_seasonal_naive.json", summary_seasonal_naive)

    # Exports
    #by_horizon_df.to_csv(run_dir / "metrics_by_horizon.csv", index=False)

    #loss_png, metrics_png = save_plots(run_dir, by_horizon_df, results)

    # Optional forecast plot
    example_naive = save_forecast_example(df, SPLIT_NAME, run_dir / "forecast_example_native.png", "Naive")
    example_seasonal_naive = save_forecast_example(df, SPLIT_NAME, run_dir / "forecast_example_seasonal_native.png", "Seasonal Naive")

    #print("\nSaved plots:", loss_png, metrics_png)
    if example_naive:
        print("Saved forecast example:", run_dir / "forecast_example.png")
    if example_seasonal_naive:
        print("Saved forecast example:", run_dir / "forecast_example.png")
    print("Saved:", run_dir / "metrics_by_horizon.csv")
    print("Run folder:", run_dir)


if __name__ == "__main__":
    main("train")
    main("val")
    main("test")