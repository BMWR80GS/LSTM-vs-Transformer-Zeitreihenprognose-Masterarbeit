# train_tft_logged.py
# -----------------------------------------------------------------------------
# Zweck dieses Skripts:
#   Training eines Temporal Fusion Transformers (TFT) mittels PyTorch Forecasting.
#   Das Modell wird auf dem vorverarbeiteten M5-Datensatz trainiert und es werden
#   zentrale Artefakte für die Auswertung und Dokumentation gespeichert.
#
# Wissenschaftlicher Kontext:
#   Der TFT kombiniert eine Encoder-Decoder-Struktur mit Attention-Mechanismen und
#   variabler Selektion. Im Vergleich zum LSTM kann der TFT komplexe Zusammenhänge
#   zwischen zeitvariierenden Kovariaten und Zielvariablen flexibler modellieren.
#
# Logging-Strategie:
#   Da PyTorch Forecasting auf PyTorch Lightning basiert, wird der CSVLogger genutzt,
#   um train_loss und val_loss zu protokollieren. Anschließend wird eine Excel-Datei
#   analog zum LSTM erzeugt.
#
# Zusätzliche Zeitreihenfeatures:
#   Für Konsistenz zum LSTM werden Lag- und Rolling-Features ebenfalls berechnet.
#   Dies ermöglicht in späteren Experimenten eine kontrollierte Untersuchung, ob
#   diese Features einen Einfluss auf die TFT-Leistung haben.
#
# Ausgabe pro Run:
#   - config.json, system_info.json, summary.json
#   - metrics.xlsx (config / epochs / summary)
#   - loss.png / metrics.png
#   - forecast_example.png
#   - best.ckpt (Checkpoint mit minimalem val_loss)
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
torch.set_float32_matmul_precision("medium")

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from run_logger import (
    get_system_info,
    save_json,
    save_excel,
    save_plots
)

# -----------------------------------------------------------------------------
# Pfade zum vorverarbeiteten Datensatz.
# -----------------------------------------------------------------------------
PROCESSED_DIR = Path("data/preprocessed")
META_PATH = PROCESSED_DIR / "meta.json"
CSV_PATH = PROCESSED_DIR / "m5_long.csv"
PARQUET_PATH = PROCESSED_DIR / "m5_long.parquet"

RUNS_DIR = "runs"

# -----------------------------------------------------------------------------
# Hyperparameter.
# -----------------------------------------------------------------------------
ENCODER_LEN = 56
PRED_LEN = 28

BATCH_SIZE = 64
MAX_EPOCHS = 5

BASE_SEED = 1
NUM_SEEDS = 1

LR = 3e-4
HIDDEN_SIZE = 32
ATTN_HEAD_SIZE = 4
HIDDEN_CONT_SIZE = 16
DROPOUT = 0.1

# -----------------------------------------------------------------------------
# Early Stopping Einstellungen für TFT (über Lightning Callback).
# -----------------------------------------------------------------------------
PATIENCE = 3

# -----------------------------------------------------------------------------
# Lag- und Rolling-Features (analog zum LSTM).
# -----------------------------------------------------------------------------
LAG_LIST = [1, 7, 28]
ROLLING_WINDOWS = [7, 28]

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def load_preprocessed():
    # Laden des vorverarbeiteten Datensatzes.
    if not META_PATH.exists():
        raise FileNotFoundError(f"{META_PATH} fehlt. Preprocessing muss zuvor ausgeführt werden.")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    if meta.get("output_format") == "parquet" and PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    else:
        raise FileNotFoundError("Keine vorverarbeitete Datei gefunden (m5_long.csv oder m5_long.parquet).")

    return df, meta


def add_time_series_features(df):
    # Ergänzung autoregressiver Merkmale (Lag und Rolling Mean) im log-space.
    # Dies erfolgt primär zur Konsistenz mit dem LSTM-Setup.
    df = df.copy()
    df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

    for lag in LAG_LIST:
        df[f"y_log_lag_{lag}"] = df.groupby("series_id")["y_log"].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"y_log_roll_mean_{w}"] = (
            df.groupby("series_id")["y_log"]
              .shift(1)
              .rolling(window=w, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    lag_cols = [f"y_log_lag_{l}" for l in LAG_LIST]
    roll_cols = [f"y_log_roll_mean_{w}" for w in ROLLING_WINDOWS]
    df[lag_cols + roll_cols] = df[lag_cols + roll_cols].fillna(0.0)

    return df


def compute_mase_denominators(train_df: pd.DataFrame, seasonality: int = 7) -> dict:
    denoms = {}
    for sid, g in train_df.groupby("series_id"):
        g = g.sort_values("time_idx")
        if "y" in g.columns:
            y = g["y"].values.astype(np.float32)
        else:
            y = np.expm1(g["y_log"].values.astype(np.float32))

        if len(y) <= seasonality:
            denoms[str(sid)] = 1.0
            continue

        diff = np.abs(y[seasonality:] - y[:-seasonality])
        den = float(np.mean(diff)) if np.mean(diff) > 0 else 1.0
        denoms[str(sid)] = den

    return denoms


def get_series_id_mapping(training_ds):
    mapping = None

    if hasattr(training_ds, "categorical_encoders"):
        encoders = getattr(training_ds, "categorical_encoders", {})
        if isinstance(encoders, dict) and "series_id" in encoders:
            enc = encoders["series_id"]
            classes = None
            if hasattr(enc, "classes_"):
                classes = list(getattr(enc, "classes_"))
            elif hasattr(enc, "classes"):
                classes = list(getattr(enc, "classes"))
            if classes is not None:
                mapping = {int(i): str(v) for i, v in enumerate(classes)}

    return mapping


def extract_series_ids_from_raw_x(raw_x, series_mapping):
    groups_key = None
    if isinstance(raw_x, dict):
        if "groups" in raw_x:
            groups_key = "groups"
        elif "group_ids" in raw_x:
            groups_key = "group_ids"

    if groups_key is None:
        raise KeyError("Konnte keine Gruppeninformation (groups/group_ids) in raw.x finden.")

    group_values = raw_x[groups_key]

    if isinstance(group_values, torch.Tensor):
        group_values = group_values.detach().cpu().numpy()

    group_values = np.asarray(group_values)

    if group_values.ndim == 2:
        group_values = group_values[:, 0]

    series_ids = []
    for v in group_values.tolist():
        try:
            v_int = int(v)
            if series_mapping is not None and v_int in series_mapping:
                series_ids.append(series_mapping[v_int])
            else:
                series_ids.append(str(v_int))
        except Exception:
            series_ids.append(str(v))

    return np.asarray(series_ids, dtype=object)


def eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, series_ids, mase_denoms):
    week_slices = [(0, 7), (7, 14), (14, 21), (21, 28)]

    pred_y = np.asarray(pred_y, dtype=np.float32)
    true_y = np.asarray(true_y, dtype=np.float32)

    pred_y = np.clip(pred_y, a_min=0.0, a_max=None)
    true_y = np.clip(true_y, a_min=0.0, a_max=None)

    abs_err = np.abs(pred_y - true_y)

    mse = float(np.mean((pred_y - true_y) ** 2))

    den = np.array([float(mase_denoms.get(str(sid), 1.0)) for sid in series_ids], dtype=np.float32)
    den = np.where(den > 0, den, 1.0)

    mae_overall = np.mean(abs_err, axis=1)
    mase = float(np.mean(mae_overall / den))

    mase_weeks = []
    for a, b in week_slices:
        mae_week = np.mean(abs_err[:, a:b], axis=1)
        mase_weeks.append(float(np.mean(mae_week / den)))

    wape_num = float(np.sum(abs_err))
    wape_den = float(np.sum(true_y))
    wape = (wape_num / wape_den) if wape_den > 0 else float("nan")

    wape_weeks = []
    for a, b in week_slices:
        num = float(np.sum(abs_err[:, a:b]))
        den_w = float(np.sum(true_y[:, a:b]))
        wape_weeks.append((num / den_w) if den_w > 0 else float("nan"))

    return {
        "mase": mase,
        "mase_w1": mase_weeks[0],
        "mase_w2": mase_weeks[1],
        "mase_w3": mase_weeks[2],
        "mase_w4": mase_weeks[3],
        "mse": mse,
        "wape": float(wape),
        "wape_w1": float(wape_weeks[0]),
        "wape_w2": float(wape_weeks[1]),
        "wape_w3": float(wape_weeks[2]),
        "wape_w4": float(wape_weeks[3]),
    }


def build_timeseries_datasets(df):
    # Aufbau von TimeSeriesDataSet-Objekten für Training/Validierung/Test.
    #
    # Split-Logik:
    #   Der maximale time_idx im Trainingssplit wird als train_cutoff interpretiert.
    #   Anschließend werden Validierung und Test jeweils um PRED_LEN erweitert.
    #
    # Hinweis:
    #   PyTorch Forecasting arbeitet typischerweise zeitbasiert mit cutoffs und
    #   generiert daraus Forecast-Fenster (predict=True).
    df = df.copy()
    df["series_id"] = df["series_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    if "y_log" not in df.columns:
        raise KeyError("Spalte 'y_log' fehlt. Preprocessing ist unvollständig.")

    train_cutoff = int(df.loc[df["split"] == "train", "time_idx"].max())
    val_cutoff = train_cutoff + PRED_LEN
    test_cutoff = val_cutoff + PRED_LEN

    # Zeitvariierende bekannte Features (auch in der Zukunft verfügbar).
    known_reals = [
        "time_idx",
        "price_s",
        "snap",
        "wday_s",
        "month_s",
        "year_s",
        "has_event_1",
        "has_event_2",
    ]

    # Zeitvariierende unbekannte Features (aus Vergangenheit bekannt, in Zukunft unbekannt).
    # Neben y_log werden hier zusätzlich Lag- und Rolling-Features aufgenommen.
    unknown_reals = ["y_log"]
    unknown_reals += [f"y_log_lag_{l}" for l in LAG_LIST]
    unknown_reals += [f"y_log_roll_mean_{w}" for w in ROLLING_WINDOWS]

    training = TimeSeriesDataSet(
        df[df.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="y_log",
        group_ids=["series_id"],

        min_encoder_length=ENCODER_LEN,
        max_encoder_length=ENCODER_LEN,
        min_prediction_length=PRED_LEN,
        max_prediction_length=PRED_LEN,

        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,

        target_normalizer=GroupNormalizer(groups=["series_id"]),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_data = df[df.time_idx <= val_cutoff]
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_data,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=train_cutoff + 1,
    )

    test_data = df[df.time_idx <= test_cutoff]
    test = TimeSeriesDataSet.from_dataset(
        training,
        test_data,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=val_cutoff + 1,
    )

    return training, validation, test


def save_forecast_example(model, test_loader, out_path):
    # Speicherung einer Beispielvorhersage zur qualitativen Plausibilisierung.
    raw = model.predict(test_loader, mode="raw", return_x=True)

    preds = raw.output.prediction.detach().cpu().numpy()
    x = raw.x
    true = x["decoder_target"].detach().cpu().numpy()

    # QuantileLoss erzeugt typischerweise mehrere Quantile.
    # Der Median wird als Punktprognose verwendet.
    if preds.ndim == 3 and preds.shape[2] > 1:
        preds = preds[:, :, preds.shape[2] // 2]
    elif preds.ndim == 3:
        preds = preds[:, :, 0]

    pred_y = np.expm1(preds).clip(min=0.0)
    true_y = np.expm1(true).clip(min=0.0)

    plt.figure()
    plt.plot(true_y[0], label="True")
    plt.plot(pred_y[0], label="Pred", linestyle="--")
    plt.title("Beispielvorhersage (TFT, Testsplit)")
    plt.xlabel("Forecast-Schritt")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df, meta = load_preprocessed()

    # Ergänzung autoregressiver Features (Lags/Rolling) analog zum LSTM.
    df = add_time_series_features(df)

    train_df = df[df["split"] == "train"].copy()
    mase_denoms = compute_mase_denominators(train_df, seasonality=7)

    config = {
        "model": "TFT",
        "encoder_len": ENCODER_LEN,
        "pred_len": PRED_LEN,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "hidden_size": HIDDEN_SIZE,
        "attn_head_size": ATTN_HEAD_SIZE,
        "hidden_cont_size": HIDDEN_CONT_SIZE,
        "dropout": DROPOUT,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lags": LAG_LIST,
        "rolling_windows": ROLLING_WINDOWS,
        "max_series": meta.get("max_series_requested"),
        "series_kept": meta.get("series_kept_after_length_filter"),
        "device": DEVICE,
        "base_seed": BASE_SEED,
        "num_seeds": NUM_SEEDS,
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    parent_run_dir = Path(RUNS_DIR) / "tft" / f"{ts}_MultiSeed_TFT__base_seed={BASE_SEED}__num_seeds={NUM_SEEDS}"
    parent_run_dir.mkdir(parents=True, exist_ok=True)

    save_json(parent_run_dir / "config.json", config)
    save_json(parent_run_dir / "system_info.json", get_system_info())

    best_overall_val_mase = float("inf")
    best_overall_val_loss = float("inf")
    best_overall_seed = None
    best_overall_ckpt = parent_run_dir / "best_overall.ckpt"

    for seed_offset in range(NUM_SEEDS):
        current_seed = int(BASE_SEED) + int(seed_offset)

        # Seed wird für Konsistenz der Experimente gesetzt.
        pl.seed_everything(current_seed, workers=False)

        run_dir = parent_run_dir / f"seed_{current_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        seed_config = dict(config)
        seed_config["seed"] = current_seed
        save_json(run_dir / "config.json", seed_config)
        save_json(run_dir / "system_info.json", get_system_info())

        training_ds, val_ds, test_ds = build_timeseries_datasets(df)

        series_mapping = get_series_id_mapping(training_ds)

        # num_workers=0 ist unter Windows häufig stabiler.
        train_loader = training_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
        val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
        test_loader = test_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

        model = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=LR,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTN_HEAD_SIZE,
            hidden_continuous_size=HIDDEN_CONT_SIZE,
            dropout=DROPOUT,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=2,
        )

        # CSVLogger wird genutzt, um pro Step/Epoche Metriken in eine Datei zu schreiben.
        csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")

        # ModelCheckpoint speichert den besten Checkpoint gemäß val_loss (val_loss = MASE im Log-Space)
        ckpt = ModelCheckpoint(
            dirpath=str(run_dir),
            filename="best",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )

        # EarlyStopping reduziert Overfitting-Risiko und spart Rechenzeit.
        early = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=DEVICE,
            devices=1,
            gradient_clip_val=0.1,
            logger=csv_logger,
            callbacks=[ckpt, early],
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

        total_start = time.time()
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        total_time = time.time() - total_start

        best_path = ckpt.best_model_path
        if best_path:
            model = TemporalFusionTransformer.load_from_checkpoint(best_path)

        save_forecast_example(model, test_loader, run_dir / "forecast_example.png")

        # Lightning erzeugt eine metrics.csv, die train_loss/val_loss enthält.
        metrics_csv = Path(run_dir) / "lightning_logs" / "version_0" / "metrics.csv"
        epoch_rows = []

        if metrics_csv.exists():
            log_df = pd.read_csv(metrics_csv)

            # Pro Epoche wird der jeweils letzte verfügbare Wert genutzt.
            # Dies ist pragmatisch, da train_loss häufig pro Step und val_loss pro Epoche geloggt wird.
            epochs = sorted(log_df["epoch"].dropna().unique().tolist())
            for ep in epochs:
                ep = int(ep)
                sub = log_df[log_df["epoch"] == ep]

                train_loss = np.nan
                val_loss = np.nan

                if "train_loss" in sub.columns:
                    tmp = sub["train_loss"].dropna()
                    if len(tmp) > 0:
                        train_loss = float(tmp.iloc[-1])

                if "val_loss" in sub.columns:
                    tmp = sub["val_loss"].dropna()
                    if len(tmp) > 0:
                        val_loss = float(tmp.iloc[-1])

                epoch_rows.append({
                    "epoch": ep + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

        # Zusätzliche Auswertung (MASE/WAPE, auch je Woche) auf Validierung und Test.
        val_raw = model.predict(val_loader, mode="raw", return_x=True)
        val_preds = val_raw.output.prediction.detach().cpu().numpy()
        val_x = val_raw.x
        val_true = val_x["decoder_target"].detach().cpu().numpy()

        if val_preds.ndim == 3 and val_preds.shape[2] > 1:
            val_preds = val_preds[:, :, val_preds.shape[2] // 2]
        elif val_preds.ndim == 3:
            val_preds = val_preds[:, :, 0]

        val_pred_y = np.expm1(val_preds).clip(min=0.0)
        val_true_y = np.expm1(val_true).clip(min=0.0)

        val_series_ids = extract_series_ids_from_raw_x(val_x, series_mapping)
        val_metrics = eval_mase_mse_wape_weekly_from_arrays(val_pred_y, val_true_y, val_series_ids, mase_denoms)

        test_raw = model.predict(test_loader, mode="raw", return_x=True)
        test_preds = test_raw.output.prediction.detach().cpu().numpy()
        test_x = test_raw.x
        test_true = test_x["decoder_target"].detach().cpu().numpy()

        if test_preds.ndim == 3 and test_preds.shape[2] > 1:
            test_preds = test_preds[:, :, test_preds.shape[2] // 2]
        elif test_preds.ndim == 3:
            test_preds = test_preds[:, :, 0]

        test_pred_y = np.expm1(test_preds).clip(min=0.0)
        test_true_y = np.expm1(test_true).clip(min=0.0)

        test_series_ids = extract_series_ids_from_raw_x(test_x, series_mapping)
        test_metrics = eval_mase_mse_wape_weekly_from_arrays(test_pred_y, test_true_y, test_series_ids, mase_denoms)

        summary = {
            "seed": current_seed,
            "best_model_path": best_path,
            "best_val_loss": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
            "total_time_sec": float(total_time),
            "epochs_ran": int(trainer.current_epoch) + 1,

            "val_mase": float(val_metrics["mase"]),
            "val_mase_w1": float(val_metrics["mase_w1"]),
            "val_mase_w2": float(val_metrics["mase_w2"]),
            "val_mase_w3": float(val_metrics["mase_w3"]),
            "val_mase_w4": float(val_metrics["mase_w4"]),

            "val_wape": float(val_metrics["wape"]),
            "val_wape_w1": float(val_metrics["wape_w1"]),
            "val_wape_w2": float(val_metrics["wape_w2"]),
            "val_wape_w3": float(val_metrics["wape_w3"]),
            "val_wape_w4": float(val_metrics["wape_w4"]),

            "test_mase": float(test_metrics["mase"]),
            "test_mase_w1": float(test_metrics["mase_w1"]),
            "test_mase_w2": float(test_metrics["mase_w2"]),
            "test_mase_w3": float(test_metrics["mase_w3"]),
            "test_mase_w4": float(test_metrics["mase_w4"]),

            "test_wape": float(test_metrics["wape"]),
            "test_wape_w1": float(test_metrics["wape_w1"]),
            "test_wape_w2": float(test_metrics["wape_w2"]),
            "test_wape_w3": float(test_metrics["wape_w3"]),
            "test_wape_w4": float(test_metrics["wape_w4"]),
        }

        save_json(run_dir / "summary.json", summary)
        save_excel(run_dir, seed_config, epoch_rows, summary)
        save_plots(run_dir, epoch_rows)

        if np.isfinite(summary["val_mase"]) and summary["val_mase"] < best_overall_val_mase:
            best_overall_val_mase = float(summary["val_mase"])
            best_overall_val_loss = float(summary["best_val_loss"]) if summary["best_val_loss"] is not None else float("inf")
            best_overall_seed = int(current_seed)
            if best_path:
                best_overall_ckpt.write_bytes(Path(best_path).read_bytes())
        elif not np.isfinite(summary["val_mase"]):
            if summary["best_val_loss"] is not None and float(summary["best_val_loss"]) < best_overall_val_loss:
                best_overall_val_loss = float(summary["best_val_loss"])
                best_overall_seed = int(current_seed)
                if best_path:
                    best_overall_ckpt.write_bytes(Path(best_path).read_bytes())

        print("Saved:", run_dir / "metrics.xlsx")
        print("Saved plots:", run_dir / "loss.png", run_dir / "metrics.png")
        print("Saved forecast example:", run_dir / "forecast_example.png")
        print("Best checkpoint:", best_path)

    save_json(parent_run_dir / "best_overall.json", {
        "best_seed": best_overall_seed,
        "best_val_mase": float(best_overall_val_mase),
        "best_val_loss": float(best_overall_val_loss),
        "best_checkpoint_path": str(best_overall_ckpt),
    })

    print("Saved multi-seed run to:", parent_run_dir)


if __name__ == "__main__":
    main()
