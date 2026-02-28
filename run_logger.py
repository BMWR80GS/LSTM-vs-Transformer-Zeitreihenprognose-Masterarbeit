# run_logger.py
# -----------------------------------------------------------------------------
# Zweck dieser Datei:
#   Diese Datei enthält Hilfsfunktionen für ein konsistentes Experiment-Tracking.
#   Pro Trainingslauf wird ein Run-Ordner angelegt, in dem zentrale Artefakte
#   abgelegt werden. Dies unterstützt Reproduzierbarkeit, Vergleichbarkeit und
#   eine nachvollziehbare Dokumentation für die wissenschaftliche Arbeit.
#
# Inhaltliche Begründung:
#   In empirischen Arbeiten mit Deep-Learning-Modellen ist die Nachvollziehbarkeit
#   der Experimente entscheidend. Dazu gehören:
#   - Die genaue Konfiguration eines Runs (Hyperparameter, Feature-Set, Gerät)
#   - Die zeitliche Entwicklung der Loss- und Metrikwerte
#   - Laufzeiten pro Epoche sowie Gesamtlaufzeit
#   - Persistierung der Ergebnisse in einem strukturierten, auswertbaren Format
#
# Artefakte pro Run:
#   - config.json: Konfiguration des Runs
#   - system_info.json: Systeminformationen (z. B. GPU, Torch-Version)
#   - metrics.xlsx: Tabellenformat (config / epochs / summary)
#   - loss.png / metrics.png: Grafische Darstellung des Trainingsverlaufs
# -----------------------------------------------------------------------------

import json
import platform
import socket
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _make_folder_safe(text):
    # Diese Funktion erzeugt einen dateisystemfreundlichen String.
    # Sonderzeichen werden ersetzt, um Probleme bei der Ordnererstellung zu vermeiden.
    text = str(text)
    out = []
    for ch in text:
        if ch.isalnum() or ch in "-_=.+":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


# def create_run_folder(base_dir, model_name, config):
#     # Pro Run wird ein separater Ordner erstellt.
#     # Der Ordnername enthält einen Zeitstempel und ausgewählte Parameter.
#     # Dies ermöglicht die schnelle Zuordnung von Ergebnissen zu Einstellungen.
#     timestamp = time.strftime("%Y%m%d-%H%M%S")

#     # Diese Parameter werden typischerweise zur Unterscheidung von Runs verwendet.
#     keys_for_name = [
#         "seed",
#         "max_series",
#         "layers",
#         "seq_len",
#         "encoder_len",
#         "horizon",
#         "pred_len",
#         "batch_size",
#         "lr",
#         "hidden_size",
#         "patience",
#     ]

#     parts = [timestamp]
#     for k in keys_for_name:
#         if k in config and config[k] is not None:
#             parts.append(f"{_make_folder_safe(k)}={_make_folder_safe(config[k])}")

#     run_name = "__".join(parts)
#     run_dir = Path(base_dir) / model_name / run_name
#     run_dir.mkdir(parents=True, exist_ok=True)
#     return run_dir


def get_system_info():
    # Es werden Systeminformationen gesammelt, die für eine Reproduktion der Ergebnisse
    # relevant sein können. Dies ist insbesondere bei GPU-Training sinnvoll.
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # Torch- und CUDA-Informationen werden ergänzt, da diese häufig die Performance
    # und das Verhalten der Modelle beeinflussen können.
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_info_error"] = str(e)

    return info


def save_json(path, data):
    # Speicherung einer Python-Datenstruktur im JSON-Format.
    # Das Format ist für Menschen lesbar und für spätere Analysen einfach nutzbar.
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# def save_excel(run_dir, config, epoch_rows, summary):
#     # Speicherung von Konfiguration, epoch-weisen Metriken und Zusammenfassung in Excel.
#     # Das Excel-Format erleichtert die Auswertung und die Integration in Berichte.
#     run_dir = Path(run_dir)
#     out_path = run_dir / "metrics.xlsx"

#     df_config = pd.DataFrame([config])
#     df_epochs = pd.DataFrame(epoch_rows)
#     df_summary = pd.DataFrame([summary])

#     with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#         df_config.to_excel(writer, sheet_name="config", index=False)
#         df_epochs.to_excel(writer, sheet_name="epochs", index=False)
#         df_summary.to_excel(writer, sheet_name="summary", index=False)

#     return out_path


def save_plots(run_dir, epoch_rows):
    # Erstellung einfacher Visualisierungen des Trainingsverlaufs.
    # Diese Plots unterstützen eine schnelle qualitative Beurteilung von Konvergenz
    # und möglichem Overfitting.
    run_dir = Path(run_dir)
    df = pd.DataFrame(epoch_rows)

    if df.empty:
        return

    # Plot 1: Loss-Verlauf
    if "train_loss" in df.columns or "val_loss" in df.columns:
        plt.figure()

        if "train_loss" in df.columns and df["train_loss"].notna().any():
            plt.plot(df["epoch"], df["train_loss"], label="train_loss")

        if "val_loss" in df.columns and df["val_loss"].notna().any():
            plt.plot(df["epoch"], df["val_loss"], label="val_loss", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss-Verlauf pro Epoche")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss.png", dpi=150)
        plt.close()

    # Plot 2: Metriken-Verlauf (MASE/MSE)
    metric_cols = ["train_mase", "val_mase", "train_mse", "val_mse"]
    if any(c in df.columns for c in metric_cols):
        plt.figure()

        if "train_mase" in df.columns and df["train_mase"].notna().any():
            plt.plot(df["epoch"], df["train_mase"], label="train_mase")

        if "val_mase" in df.columns and df["val_mase"].notna().any():
            plt.plot(df["epoch"], df["val_mase"], label="val_mase", linestyle="--")

        if "train_mse" in df.columns and df["train_mse"].notna().any():
            plt.plot(df["epoch"], df["train_mse"], label="train_mse")

        if "val_mse" in df.columns and df["val_mse"].notna().any():
            plt.plot(df["epoch"], df["val_mse"], label="val_mse", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Wert")
        plt.title("MASE / MSE pro Epoche")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "metrics.png", dpi=150)
        plt.close()
