# train_lstm_logged.py
# -----------------------------------------------------------------------------
# Ziel:
#   Training eines LSTM-Modells für Multi-Horizon Forecasting auf dem
#   vorverarbeiteten M5-Datensatz (Long-Format).
#
# Wesentliche Punkte dieser Version:
#   (1) Autoregressive Zeitreihenfeatures werden aus y_z abgeleitet (nicht aus y_log),
#       um eine konsistente Skalierung der Eingangsvariablen sicherzustellen.
#       Dies reduziert Optimierungsprobleme durch gemischte Feature-Skalen.
#
#   (2) Early Stopping wird implementiert, um unnötige Epochen zu vermeiden und
#       Overfitting-Risiko zu reduzieren. Überwacht wird val_mase, da diese Metrik
#       für den Modellvergleich relevant ist.
#
#   (3) Pro Run werden Metriken, Laufzeiten und Plots in einem Run-Ordner gespeichert.
#
# Voraussetzungen:
#   - data/preprocessed/meta.json
#   - data/preprocessed/m5_long.csv oder m5_long.parquet
#   - Spalten: series_id, time_idx, split, y, y_log, y_z sowie exogene Features
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from run_logger import (
    get_system_info,
    save_json,
    save_plots
)

# -----------------------------------------------------------------------------
# Globale Konfiguration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 1
NUM_SEEDS = 3
MAX_SERIES = 200

SEQ_LEN = 56
HORIZON = 28

BATCH_SIZE = 512
MAX_EPOCHS = 10


LR = 3e-3
LR_SCHEDULER = "plateau"   # nur für Logging/Config
LR_FACTOR = 0.5            # LR wird mit diesem Faktor multipliziert
LR_PATIENCE = 5            # Epochen ohne Verbesserung bis Reduktion
LR_MIN = 1e-5              # Untergrenze


# -----------------------------------------------------------------------------
# Modell-Architektur Konfiguration
# -----------------------------------------------------------------------------
# Hidden_Size = Anzahl der versteckten Einheiten pro Layer.
# LAYER = Anzahl der LSTM-Layer.
# DROPOUT = Dropout zwischen den LSTM-Layern (wirkt nur bei num_layers > 1).
HIDDEN_SIZE = 128
LAYER = 2
DROPOUT = 0.2

# -----------------------------------------------------------------------------
# Embedding Konfiguration (statische kategoriale Merkmale)
# -----------------------------------------------------------------------------

# Embedding-Dimensionen für Merkmale, wie Item, Store, State.
# Die größe der Dimensionen richtet sich grob nach der Anzahl der Kategorien pro Merkmal.
ITEM_EMB_DIM = 8
STORE_EMB_DIM = 4
STATE_EMB_DIM = 2

# -----------------------------------------------------------------------------
# Early Stopping Konfiguration
# -----------------------------------------------------------------------------
# PATIENCE = Anzahl aufeinanderfolgender Epochen ohne Verbesserung.
# MIN_DELTA = Mindestverbesserung der Metrik, damit es als echte Verbesserung zählt.
PATIENCE = 8
MIN_DELTA = 0.001


# -----------------------------------------------------------------------------
# Pfade
# -----------------------------------------------------------------------------
PREP_DIR = Path("data") / "preprocessed"
CSV_PATH = PREP_DIR / "m5_long.csv"
PARQUET_PATH = PREP_DIR / "m5_long.parquet"
META_PATH = PREP_DIR / "meta.json"

RUNS_DIR = Path("runs") / "lstm"


def set_seed(seed: int) -> None:
    # Setzen der Zufallsseeds zur Erhöhung der Reproduzierbarkeit.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preprocessed():
    # Laden des vorverarbeiteten Datensatzes.
    # Parquet wird bevorzugt, da es i. d. R. schneller und speichereffizienter ist.
    if not META_PATH.exists():
        raise FileNotFoundError(f"{META_PATH} fehlt. Zuerst preprocess_data.py ausführen.")

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    else:
        raise FileNotFoundError("Keine preprocessed Datei gefunden (m5_long.csv oder m5_long.parquet).")

    return df


def add_autoregressive_features_from_yz(df: pd.DataFrame) -> pd.DataFrame:
    # Erzeugung autoregressiver Features aus y_z.
    #
    # Hintergrund:
    #   - Das Modell wird im log-space trainiert (y_log), da dies Ausreißer reduziert.
    #   - Für autoregressive Eingänge ist jedoch y_z vorteilhaft, da es pro Serie z-standardisiert ist
    #     und somit eine konsistentere Skala besitzt.
    #
    # Hinweis:
    #   Dieser Schritt ist bewusst im Trainingsskript enthalten, damit experimentell nachvollziehbar
    #   bleibt, welche Feature-Sets im jeweiligen Run genutzt wurden.

    df = df.copy()

    LAG_LIST = [7, 14, 28]
    ROLLING_WINDOWS = [7, 28]

    for lag in LAG_LIST:
        df[f"y_z_lag_{lag}"] = df.groupby("series_id")["y_z"].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"y_z_roll_mean_{w}"] = (
            df.groupby("series_id")["y_z"].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
        )
        df[f"y_z_roll_std_{w}"] = (
            df.groupby("series_id")["y_z"].shift(1).rolling(w).std().reset_index(level=0, drop=True)
        )

    # Für die ersten Tage je Serie entstehen NaNs durch shift/rolling.
    # Diese werden mit 0 gefüllt, da die Eingänge z-standardisiert sind und 0 dem Serienmittel entspricht.
    fill_cols = [c for c in df.columns if c.startswith("y_z_lag_") or c.startswith("y_z_roll_")]
    df[fill_cols] = df[fill_cols].fillna(0.0).astype(np.float32)

    return df


def build_feature_columns():
    # Zusammenstellung der Feature-Spalten.
    #
    # Enthalten sind:
    #   - y_z (als primärer autoregressiver Input)
    #   - Lags und Rolling-Statistiken (aus y_z abgeleitet)
    #   - Exogene Features (Kalender, Preis, SNAP, Events, ...)
    #
    # Wichtig:
    #   Die Zielvariable y_log wird nicht als Feature verwendet, um Leckagen zu vermeiden.

    EXOG_FEATURES = [
        "price_s",
        "price_missing",
        "snap",
        "wday_s",
        "month_s",
        "year_s",
        "has_event_1",
        "has_event_2",
    ]

    LAG_LIST = [7, 14, 28]
    ROLLING_WINDOWS = [7, 28]

    feature_cols = ["y_z"]
    feature_cols += [f"y_z_lag_{l}" for l in LAG_LIST]
    feature_cols += [f"y_z_roll_mean_{w}" for w in ROLLING_WINDOWS]
    feature_cols += [f"y_z_roll_std_{w}" for w in ROLLING_WINDOWS]

    feature_cols += EXOG_FEATURES

    return feature_cols


def compute_mase_denominators(train_df: pd.DataFrame, seasonality: int = 7) -> dict:
    # Berechnung des MASE-Denominators pro Serie auf Basis der Trainingsdaten.
    # Dient der Berechung der MASE Metrik für das Training und die Validierung. Entspricht dem Nenner der MASE-Formel, der den durchschnittlichen Fehler eines Naive-Forecasters mit saisonalem Lag darstellt.
    #
    # Definition:
    #   Denominator = mean(|y_t - y_{t-seasonality}|)
    #
    # Hintergrund:
    #   MASE skaliert den absoluten Fehler am typischen Fehler eines Naive-Forecasters.
    #   Dies ermöglicht einen vergleichbaren Maßstab über Serien mit sehr unterschiedlichen Niveaus.
    denoms = {}

    for series_id, series_values in train_df.groupby("series_id"):
        series_values = series_values.sort_values("time_idx")
        y = series_values["y"].values.astype(np.float32)

        # Für sehr kurze Serien (länger als die Saisonalität) wird der Denominator auf 1 gesetzt, um Division durch Null zu vermeiden.
        if len(y) <= seasonality:
            denoms[series_id] = 1.0
            continue

        #g

        # Berechnung des durchschnittlichen absoluten Unterschieds zwischen y_t (aktueller Tag) und y_{t-seasonality} (7 Tage zurück).
        diff = np.abs(y[seasonality:] - y[:-seasonality])
        den = float(np.mean(diff)) if np.mean(diff) > 0 else 1.0
        denoms[series_id] = den

    return denoms


def build_windows(df: pd.DataFrame, split_name: str, series_to_idx: dict, feature_cols: list):
    # Erzeugung von Sliding-Window Samples.
    #
    # Definition:
    #   Für jeden Startzeitpunkt t wird ein Sample erzeugt:
    #     X = Features der Zeitpunkte [t-SEQ_LEN, ..., t-1]
    #     Y = y_log der Zeitpunkte   [t, ..., t+HORIZON-1]
    #
    # Split-Logik:
    #   Ein Sample wird erzeugt, wenn split[t] == split_name.
    feature_list, y_log_list, series_list, item_list, store_list, state_list = [], [], [], [], [], []

    for series_id, series_values in df.groupby("series_id"):
        series_values = series_values.sort_values("time_idx").reset_index(drop=True)

        X_feat = series_values[feature_cols].astype(np.float32).values
        y_log = series_values["y_log"].astype(np.float32).values
        splits = series_values["split"].values

        item_code = int(series_values["item_id_code"].iloc[0])
        store_code = int(series_values["store_id_code"].iloc[0])
        state_code = int(series_values["state_id_code"].iloc[0])

        # Erzeugung von Samples für die Zeitpunkte, an denen split == split_name gilt. Zum Beispiel "train" oder "val".
        # Auswahl in einer Range zwischen SEQ_LEN = 56 (um genügend Vergangenheit für die Features zu haben) und len(series_values) - HORIZON + 1 (um genügend Zukunft für die Targets zu haben).
        # Vorgehen: Für jede Serie werden die Tage ab 56 (Seq_len) bis zu der Anzahl der Tage der Serie minus der Horizon durchlaufen. Es wird geprüft, ob der Split an diesem Tag mit dem gewünschten split_name übereinstimmt. Wenn ja, wird ein Sample erzeugt.
        for day in range(SEQ_LEN, len(series_values) - HORIZON + 1):
            # Wenn der an dem ausgewählten Tag (day) der zugehörige Split nicht mit dem gewünschten Split übereinstimmt, wird dieses Sample übersprungen. 
            if splits[day] != split_name:
                continue

            # Erzeugung eines Samples:
            # feature_list: Die Features der vorherigen Tage werden der Liste hinzugefügt.
            feature_list.append(X_feat[day - SEQ_LEN : day])
            # y_log_list: Die Zielwerte (y_log) der nächsten 28 Tage (HORIZON) werden der Liste hinzugefügt.
            y_log_list.append(y_log[day : day + HORIZON])
            # series_list: Der Index der Serie wird der Liste hinzugefügt.
            series_list.append(series_to_idx[series_id])
            # item_list, store_list, state_list: Die Codes für Item, Store und State der Serie werden der jeweiligen Liste hinzugefügt. Diese Codes werden später für die Embedding-Layer benötigt.
            item_list.append(item_code)
            store_list.append(store_code)
            state_list.append(state_code)

    # Wenn feature_list leer ist, bedeutet dies, dass kein Sample für den angegebenen split_name gefunden wurde. In diesem Fall werden None-Werte zurückgegeben, um anzuzeigen, dass keine Daten vorhanden sind.
    if len(feature_list) == 0:
        return None, None, None, None, None, None

    # Die Listen werden in numpy-Arrays umgewandelt. feature_list und y_log_list werden zu 3D-Arrays (Anzahl_Samples, SEQ_LEN, Anzahl_Features) bzw. (Anzahl_Samples, HORIZON). series_list, item_list, store_list und state_list werden zu 1D-Arrays mit den entsprechenden Indizes.
    # np.stack wird verwendet, um die Listen von Arrays entlang einer neuen Achse zu stapeln, wodurch ein 3D-Array entsteht. np.array wird verwendet, um die Listen von Indizes in 1D-Arrays umzuwandeln.
    # n.stack = 0 bedeutet, dass die Arrays entlang der ersten Achse (der Sample-Achse) gestapelt werden. Dadurch entsteht ein Array der Form (Anzahl_Samples, SEQ_LEN, Anzahl_Features) für feature_list und (Anzahl_Samples, HORIZON) für y_log_list.
    return (
        np.stack(feature_list, 0),
        np.stack(y_log_list, 0),
        np.array(series_list, dtype=np.int64),
        np.array(item_list, dtype=np.int64),
        np.array(store_list, dtype=np.int64),
        np.array(state_list, dtype=np.int64),
    )


class Many_to_One_LSTM(nn.Module):
    # Many-to-One LSTM-Architektur für Multi-Horizon Forecasting.
    # Der letzte Hidden-State wird als Sequenzrepräsentation genutzt. Dies entspricht damit dem Many-to-One Ansatzes.
    def __init__(self, n_features: int, hidden_size: int, horizon: int, num_items: int, num_stores: int, num_states: int):
        
        super().__init__()
        
        # Aufbau der Embedding-Layer für die statischen kategorialen Merkmale (Item, Store, State).
        # Durch diese Embeddings soll das Modell besser verstehen lernen, wie sich verschiedene Items, Stores und States auf die Verkaufszahlen auswirken, ohne dass diese Informationen explizit über Features mitgeliefert werden.
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=ITEM_EMB_DIM)
        self.store_emb = nn.Embedding(num_embeddings=num_stores, embedding_dim=STORE_EMB_DIM)
        self.state_emb = nn.Embedding(num_embeddings=num_states, embedding_dim=STATE_EMB_DIM)

        input_size = n_features + ITEM_EMB_DIM + STORE_EMB_DIM + STATE_EMB_DIM

        # Wenn Anzahl der versteckten Layer == 1 ist, dann wird kein Dropout angewendet, da Dropout nur zwischen Layern wirkt. Wenn LAYER > 1, wird Dropout zwischen den LSTM-Layern aktiviert.
        if LAYER == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=LAYER,
                dropout=DROPOUT,
                batch_first=True,
            )

        # Das lineare Layer bildet den letzten Hidden-State der LSTM auf die HORIZON-Dimension ab, um die Vorhersage der nächsten HORIZON Zeitschritte zu ermöglichen.
        self.fc = nn.Linear(hidden_size, horizon)

    # Forward-Pass des Modells
    def forward(self, x, item_idx, store_idx, state_idx):
        # Die einzelnen Embedding-Vektoren für Item, Store und State werden mit dem jeweiligen Index aus den Eingabedaten abgerufen. Diese Vektoren repräsentieren die Informationen über die statischen Merkmale der Serie.
        # Zum Beispiel Item_idx == 1. Holt aus der Matrix die bei der Erstellung der Embedding-Layer mit num_items definiert wurde, die Zeile mit Index 1, welche den Embedding-Vektor für dieses Item enthält. Das gleiche gilt für Store_idx und State_idx.
        item_vec = self.item_emb(item_idx)
        store_vec = self.store_emb(store_idx)
        state_vec = self.state_emb(state_idx)

        # Die abgerufenen Embedding-Vektoren werden entlang der letzten Dimension (embedding_dim) zu einem einzigen Vektor pro Sample zusammengeführt. Dieser Vektor enthält die Informationen über Item, Store und State der Serie.
        static_vec = torch.cat([item_vec, store_vec, state_vec], dim=-1)
        # Der statische Vektor wird nun so angepasst, dass er die gleiche Sequenzlänge (Tage) wie die Eingabesequenz x hat. Dies wird erreicht, indem der Vektor entlang der Sequenzdimension (dim=1) dupliziert wird. Dadurch entsteht ein Tensor, der für jeden Zeitschritt die gleichen statischen Informationen enthält.
        static_seq = static_vec.unsqueeze(1).expand(-1, x.size(1), -1)
        # Die Informationen aus den Vektoren werden nun an die Zeitreiheninformationen angehängt. Dadurch erhält jeder Zeitschritt der Eingabesequenz x zusätzlich die Informationen über Item, Store und State der Serie.
        x = torch.cat([x, static_seq], dim=-1)

        # LSTM Vorwärtsdurchlauf
        out, _ = self.lstm(x)
        # Nutzung des letzten Hidden-States für die Vorhersage (1-56, hier wird nur der letzte Zeitschritt genutzt, der die gesamte Sequenz repräsentiert) - Einschränkung ersten Tage der Serie werden kaum berücksichtigt. Many-to-One Ansatz.
        last = out[:, -1, :]
        # Der Hiddenstate mit allen Hiden-Units wird auf die Horizon-Dimension abgebildet. Foward pass durch das lineare Layer.
        out_final = self.fc(last)
        # Ausgabe: Vorhersage der nächsten HORIZON Zeitschritte im log-space.
        return out_final


def eval_loss_logspace(model: nn.Module, loader: DataLoader) -> float:
    # Berechnung des MSE-Loss im log-space für Validation.
    model.eval()
    losses = []
    # Funktion zur Ermittlung des MSE-Losses im log-space, da das Modell im log-space trainiert wird. 
    loss_function = nn.MSELoss()

    # torch.no_grad() wird genutzt, um die Berechnung der Vorhersagen und des Losses durchzuführen, ohne dass dabei Gradienten berechnet oder gespeichert werden. Dies spart Speicher und Rechenzeit, da wir uns im Evaluierungsmodus befinden und keine Backpropagation durchführen müssen.
    with torch.no_grad():
        for feature_values, y_log_actual, _series_id, item_idx, store_idx, state_idx in loader:
            feature_values = feature_values.to(DEVICE)
            y_log_actual = y_log_actual.to(DEVICE)
            item_idx = item_idx.to(DEVICE)
            store_idx = store_idx.to(DEVICE)
            state_idx = state_idx.to(DEVICE)

            # Auf Basis der Eingabeinformationen errechnet das Modell die Vorhersagen im log-space. Diese Vorhersagen werden dann mit den tatsächlichen Zielwerten (y_log_actual) verglichen, um den MSE-Loss zu berechnen. Die berechneten Losses werden in einer Liste gesammelt, um am Ende den Durchschnitts-Loss über alle Batches zu ermitteln.
            pred_log = model(feature_values, item_idx, store_idx, state_idx)
            losses.append(loss_function(pred_log, y_log_actual).item())

    return float(np.mean(losses)) if losses else float("nan")


def eval_mase_mse(model: nn.Module, loader: DataLoader, idx_to_series: dict, mase_denoms: dict):
    # Berechnung von MASE und MSE im Originalraum.
    #
    # Vorgehen:
    #   - Vorhersagen liegen im log-space vor -> Rücktransformation via expm1.
    #   - MSE wird im Originalraum berechnet (Sales-Einheiten).
    #   - MASE wird pro Sample normalisiert anhand des Denominators der jeweiligen Serie.
    model.eval()
    all_mase, all_mse = [], []

    # torch.no_grad() wird genutzt, um die Berechnung der Vorhersagen und des Losses durchzuführen, ohne dass dabei Gradienten berechnet oder gespeichert werden. Dies spart Speicher und Rechenzeit, da wir uns im Evaluierungsmodus befinden und keine Backpropagation durchführen müssen.
    with torch.no_grad():
        for feature_values, y_log_actual, series_id_idx, item_idx, store_idx, state_idx in loader:
            feature_values = feature_values.to(DEVICE)
            y_log_actual = y_log_actual.to(DEVICE)
            item_idx = item_idx.to(DEVICE)
            store_idx = store_idx.to(DEVICE)
            state_idx = state_idx.to(DEVICE)

            # Vorhersage des Modells erfolgt immer im Log-Raum
            pred_log = model(feature_values, item_idx, store_idx, state_idx)

            # Vorhersage und IST Werte werden mittels expm1 zurück in den Originalraum transformiert. clamp_min(0.0) stellt sicher, dass negative Vorhersagen (die im Originalraum keinen Sinn ergeben würden) auf 0 gesetzt werden.
            pred_y = torch.expm1(pred_log).clamp_min(0.0)
            true_y = torch.expm1(y_log_actual).clamp_min(0.0)

            # Anwenden der Formel für MSE im Originalraum: mean((y - y_hat)^2) und Hinzufügen zum all_mse Liste.
            all_mse.append(torch.mean((pred_y - true_y) ** 2).item())

            # Berechnung des MASE
            mae_native = []
            # Für die Berechnung der MASE wird für jedes Sample der entsprechende Denominator aus mase_denoms anhand der Serien-ID ermittelt. Dieser Denominator repräsentiert den durchschnittlichen Fehler eines Naive-Forecasters für die jeweilige Serie und dient als Normalisierungsfaktor für den absoluten Fehler (MAE) des Modells. Wenn für eine Serie kein Denominator gefunden wird, wird standardmäßig 1.0 verwendet, um Division durch Null zu vermeiden.
            for i in series_id_idx.cpu().numpy().tolist():
                # Ermittlen der Series_id anhand des Indexes aus dem Dictory idx_to_series.
                series_id = idx_to_series[i]
                # Ermiteln des MAE Loss Value für die Serie aus mase_denoms. Wenn kein Denominator gefunden wird, wird 1.0 verwendet.
                mae_native.append(mase_denoms.get(series_id, 1.0))
            # Die Liste der Denominator-Werte wird in einen Tensor umgewandelt, damit sie für die Berechnung des MASE mit den Vorhersagen und den tatsächlichen Werten kompatibel ist. Der Tensor wird auf das gleiche Gerät wie pred_y verschoben, um sicherzustellen, dass die Berechnung auf der GPU (falls verfügbar) durchgeführt wird. Das unsqueeze(1) fügt eine zusätzliche Dimension hinzu, damit die Form des Denominator-Tensors mit der Form der MAE-Berechnung übereinstimmt.
            mae_native = torch.tensor(mae_native, dtype=torch.float32, device=pred_y.device).unsqueeze(1)

            # MAE mittles Formel berechnen
            mae = torch.mean(torch.abs(pred_y - true_y), dim=1, keepdim=True)
            # MASE berechnen, indem der MAE durch den MAE der nativen Vorhersage geteilt wird. Das mean() am Ende berechnet den Durchschnitts-MASE über alle Samples im Batch, und item() extrahiert den Wert als Python-Float.
            mase = (mae / mae_native).mean().item()

            all_mase.append(mase)

    return float(np.mean(all_mase)), float(np.mean(all_mse))

def eval_mase_mse_wape_weekly(model: nn.Module, loader: DataLoader, idx_to_series: dict, mase_denoms: dict):
    # Berechnung von MASE, MSE und WAPE im Originalraum, zusätzlich aufgeteilt in Wochen (7-Tage Blöcke).
    #
    # Definition WAPE:
    #   WAPE = sum(|y - y_hat|) / sum(y)
    #
    # Wochen-Logik (HORIZON=28):
    #   Woche 1 = Tage 1-7, Woche 2 = Tage 8-14, Woche 3 = Tage 15-21, Woche 4 = Tage 22-28
    model.eval()

    week_slices = [(0, 7), (7, 14), (14, 21), (21, 28)]

    all_mse = []
    all_mase_overall = []
    all_mase_weeks = [[] for _ in range(4)]

    wape_num_overall = 0.0
    wape_den_overall = 0.0
    wape_num_weeks = [0.0, 0.0, 0.0, 0.0]
    wape_den_weeks = [0.0, 0.0, 0.0, 0.0]

    with torch.no_grad():
        for feature_values, y_log_actual, series_id_idx, item_idx, store_idx, state_idx in loader:
            feature_values = feature_values.to(DEVICE)
            y_log_actual = y_log_actual.to(DEVICE)
            item_idx = item_idx.to(DEVICE)
            store_idx = store_idx.to(DEVICE)
            state_idx = state_idx.to(DEVICE)

            pred_log = model(feature_values, item_idx, store_idx, state_idx)

            # Rücktransformation in den Originalraum (Sales)
            pred_y = torch.expm1(pred_log).clamp_min(0.0)
            true_y = torch.expm1(y_log_actual).clamp_min(0.0)

            abs_err = torch.abs(pred_y - true_y)

            # MSE (gesamt)
            all_mse.append(torch.mean((pred_y - true_y) ** 2).item())

            # Denominators je Sample (MASE)
            den_values = []
            for i in series_id_idx.cpu().numpy().tolist():
                series_id = idx_to_series[i]
                den_values.append(mase_denoms.get(series_id, 1.0))
            den = torch.tensor(den_values, dtype=torch.float32, device=pred_y.device).unsqueeze(1)

            # MASE (gesamt)
            mae_overall = torch.mean(abs_err, dim=1, keepdim=True)
            all_mase_overall.append((mae_overall / den).mean().item())

            # MASE je Woche
            for w, (a, b) in enumerate(week_slices):
                mae_week = torch.mean(abs_err[:, a:b], dim=1, keepdim=True)
                all_mase_weeks[w].append((mae_week / den).mean().item())

            # WAPE (gesamt)
            wape_num_overall += float(abs_err.sum().item())
            wape_den_overall += float(true_y.sum().item())

            # WAPE je Woche
            for w, (a, b) in enumerate(week_slices):
                wape_num_weeks[w] += float(abs_err[:, a:b].sum().item())
                wape_den_weeks[w] += float(true_y[:, a:b].sum().item())

    mase = float(np.mean(all_mase_overall)) if all_mase_overall else float("nan")
    mse = float(np.mean(all_mse)) if all_mse else float("nan")

    mase_week_values = []
    for w in range(4):
        mase_week_values.append(float(np.mean(all_mase_weeks[w])) if all_mase_weeks[w] else float("nan"))

    wape = (wape_num_overall / wape_den_overall) if wape_den_overall > 0 else float("nan")
    wape_week_values = []
    for w in range(4):
        wape_week_values.append((wape_num_weeks[w] / wape_den_weeks[w]) if wape_den_weeks[w] > 0 else float("nan"))

    return {
        "mase": mase,
        "mase_w1": mase_week_values[0],
        "mase_w2": mase_week_values[1],
        "mase_w3": mase_week_values[2],
        "mase_w4": mase_week_values[3],
        "mse": mse,
        "wape": float(wape),
        "wape_w1": float(wape_week_values[0]),
        "wape_w2": float(wape_week_values[1]),
        "wape_w3": float(wape_week_values[2]),
        "wape_w4": float(wape_week_values[3]),
    }


def ensure_run_dir() -> Path:
    # Erzeugung eines Run-Verzeichnisses mit den wichtigsten Einstellungen im Namen.
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = (f"{ts}_with_Embedding__seed={SEED}__max_series={MAX_SERIES}__seq_len={SEQ_LEN}__horizon={HORIZON}__batch_size={BATCH_SIZE}__lr={LR}__hidden_size={HIDDEN_SIZE}__patience={PATIENCE}")
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def get_system_info() -> dict:
    # Sammlung einfacher Systeminformationen für die Dokumentation der Experimente.
    return {
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def build_seed_list() -> list:
    # Wenn NUM_SEEDS = 1, wird einfach der Wert aus SEED genutzt.
    # Wenn NUM_SEEDS > 1, werden fortlaufende Seeds genutzt (SEED, SEED+1, ...).
    if NUM_SEEDS <= 1:
        return [SEED]
    return [SEED + i for i in range(NUM_SEEDS)]


def reorder_metrics_columns(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    # Spaltenreihenfolge: Wochen-Metriken direkt neben val_mase / val_wape.
    column_order = [
        "epoch",
        "train_loss",
        "val_loss",

        "val_mase",
        "val_mase_w1",
        "val_mase_w2",
        "val_mase_w3",
        "val_mase_w4",

        "val_wape",
        "val_wape_w1",
        "val_wape_w2",
        "val_wape_w3",
        "val_wape_w4",

        "val_mse",
        "lr",
        "epoch_time_sec",
    ]

    existing_columns = [c for c in column_order if c in metrics_dataframe.columns]
    remaining_columns = [c for c in metrics_dataframe.columns if c not in existing_columns]
    return metrics_dataframe[existing_columns + remaining_columns]


def train_one_seed(
    df: pd.DataFrame,
    feature_cols: list,
    series_to_idx: dict,
    idx_to_series: dict,
    item_ids: list,
    store_ids: list,
    state_ids: list,
    mase_denoms: dict,
    run_dir: Path,
    seed_value: int) -> dict:
    
    set_seed(seed_value)

    # Erzeugung der Sliding Windows über die Funktion build_windows.
    Feature_train, Y_log_train, Series_train, Item_train, Store_train, State_train = build_windows(df, "train", series_to_idx, feature_cols)
    Feature_val, Y_log_val, Series_val, Item_val, Store_val, State_val = build_windows(df, "val", series_to_idx, feature_cols)

    # Aufbau von je einem TensorDataset für Training und Validierung.
    train_ds = TensorDataset(
        torch.tensor(Feature_train, dtype=torch.float32),
        torch.tensor(Y_log_train, dtype=torch.float32),
        torch.tensor(Series_train, dtype=torch.long),
        torch.tensor(Item_train, dtype=torch.long),
        torch.tensor(Store_train, dtype=torch.long),
        torch.tensor(State_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(Feature_val, dtype=torch.float32),
        torch.tensor(Y_log_val, dtype=torch.float32),
        torch.tensor(Series_val, dtype=torch.long),
        torch.tensor(Item_val, dtype=torch.long),
        torch.tensor(Store_val, dtype=torch.long),
        torch.tensor(State_val, dtype=torch.long),
    )

    # Aufbau von DataLoadern für Training und Validierung. Der Trainings-DataLoader wird mit shuffle=True erstellt, um die Reihenfolge der Samples in jedem Epochendurchlauf zu randomisieren, was zu einem robusteren Training führen soll.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Modellinitialisierung.
    n_features = Feature_train.shape[-1]
    # Das Modell wird mit den entsprechenden Hyperparametern und der Anzahl der eindeutigen Items, Stores und States initialisiert.
    model = Many_to_One_LSTM(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        horizon=HORIZON,
        num_items=len(item_ids),
        num_stores=len(store_ids),
        num_states=len(state_ids),
    ).to(DEVICE)

    # Optimizer, Loss-Funktion und Scheduler Setup.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.MSELoss()

     # ReduceLROnPlateau Scheduler
    # Steuerung der LearningRate anhand val_mase. 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )

    # Protokollierung pro Epoche.
    epoch_rows = []

    # Early Stopping State.
    best_val_mase = float("inf")
    best_epoch = -1
    no_improve = 0
    best_model_path = run_dir / "best_model.pt"

    total_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()

        # Training (MSE im log-space).
        model.train()
        train_losses = []

        # non_blocking ist notwendig um den Vorteil von pin_memory=True auszuspielen
        for feature_values, y_log_actual, _series_id, item_idx, store_idx, state_idx in train_loader:
            feature_values = feature_values.to(DEVICE, non_blocking=True)
            y_log_actual = y_log_actual.to(DEVICE, non_blocking=True)
            item_idx = item_idx.to(DEVICE, non_blocking=True)
            store_idx = store_idx.to(DEVICE, non_blocking=True)
            state_idx = state_idx.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_log = model(feature_values, item_idx, store_idx, state_idx)
            loss = loss_function(pred_log, y_log_actual)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Validation (Loss im log-space, MASE/MSE/WAPE im Originalraum).
        val_loss = eval_loss_logspace(model, val_loader)
        val_metrics = eval_mase_mse_wape_weekly(model, val_loader, idx_to_series, mase_denoms)

        val_mse = val_metrics["mse"]

        val_mase = val_metrics["mase"]
        val_mase_w1 = val_metrics["mase_w1"]
        val_mase_w2 = val_metrics["mase_w2"]
        val_mase_w3 = val_metrics["mase_w3"]
        val_mase_w4 = val_metrics["mase_w4"]

        val_wape = val_metrics["wape"]
        val_wape_w1 = val_metrics["wape_w1"]
        val_wape_w2 = val_metrics["wape_w2"]
        val_wape_w3 = val_metrics["wape_w3"]
        val_wape_w4 = val_metrics["wape_w4"]

        # Scheduler step basierend auf val_mase.
        if np.isfinite(val_mase):
            scheduler.step(val_mase)

        # Messung der Epochezeit.
        epoch_time = time.perf_counter() - t0

        # Early Stopping basiert auf val_mase.
        improved = (best_val_mase - val_mase) > MIN_DELTA
        if improved:
            best_val_mase = val_mase
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1

        # Konsolenausgabe für die Übersicht beim Training.
        print(
            f"Seed {seed_value} | "
            f"Epoch {epoch:02d} | "
            f"train_loss (MSE Log-Space)={train_loss:.4f} val_loss (MSE Log-Space)={val_loss:.4f} | "
            f"val_mase={val_mase:.4f} (w1={val_mase_w1:.4f}, w2={val_mase_w2:.4f}, w3={val_mase_w3:.4f}, w4={val_mase_w4:.4f}) val_mse={val_mse:.4f} | "
            f"val_wape={val_wape:.4f} (w1={val_wape_w1:.4f}, w2={val_wape_w2:.4f}, w3={val_wape_w3:.4f}, w4={val_wape_w4:.4f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={epoch_time:.1f}s | no_improve={no_improve}"
        )

        # Informationen über die Trainingsepoche speichern.
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mase": val_mase,
                "val_mse": val_mse,
                "val_mase_w1": val_mase_w1,
                "val_mase_w2": val_mase_w2,
                "val_mase_w3": val_mase_w3,
                "val_mase_w4": val_mase_w4,
                "val_wape": val_wape,
                "val_wape_w1": val_wape_w1,
                "val_wape_w2": val_wape_w2,
                "val_wape_w3": val_wape_w3,
                "val_wape_w4": val_wape_w4,
                "epoch_time_sec": epoch_time,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        # Wenn die Anzhal der Epochen ohne Verbesserung die Patience überschreitet, wird das Training abgebrochen.
        if no_improve >= PATIENCE:
            break

    total_time = time.perf_counter() - total_start

    metrics_dataframe = pd.DataFrame(epoch_rows)
    metrics_dataframe = reorder_metrics_columns(metrics_dataframe)
    metrics_dataframe.to_csv(run_dir / "metrics.csv", index=False)

    save_plots(run_dir, epoch_rows)

    # Speichern der wichtigsten Metriken pro Seed.
    summary = {
        "seed": int(seed_value),
        "best_val_mase": float(best_val_mase),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(epoch_rows)),
        "total_time_sec": float(total_time),
        "mean_epoch_time_sec": float(np.mean([r["epoch_time_sec"] for r in epoch_rows])) if epoch_rows else float("nan"),
        "best_model_path": str(best_model_path),
        "n_train_samples": int(len(train_ds)),
        "n_val_samples": int(len(val_ds)),
        "n_features": int(n_features),
    }

    return summary


def main():
    # Dataframe Laden.
    df = load_preprocessed()

    # Ergänzung der autoregressiven Features auf konsistenter Skala (y_z-basiert).
    df = add_autoregressive_features_from_yz(df)

    # Zusammenstellung der finalen Feature-Liste.
    feature_cols = build_feature_columns()

    # Plausibilitätsprüfung: Features dürfen keine NaNs enthalten.
    # NaNs würden im Training typischerweise zu instabilen Loss-Werten führen.
    if df[feature_cols].isna().values.any():
        raise RuntimeError("NaN-Werte in Feature-Spalten gefunden. Bitte prüfen.")

    # Mapping series_id -> fortlaufender Index.
    series_ids = sorted(df["series_id"].unique())
    series_to_idx = {series_id: i for i, series_id in enumerate(series_ids)}
    idx_to_series = {i: series_id for series_id, i in series_to_idx.items()}

    item_ids = sorted(df["item_id"].unique())
    store_ids = sorted(df["store_id"].unique())
    state_ids = sorted(df["state_id"].unique())

    item_to_idx = {item: i for i, item in enumerate(item_ids)}
    store_to_idx = {store: i for i, store in enumerate(store_ids)}
    state_to_idx = {state: i for i, state in enumerate(state_ids)}

    df["item_id_code"] = df["item_id"].map(item_to_idx).astype(np.int64)
    df["store_id_code"] = df["store_id"].map(store_to_idx).astype(np.int64)
    df["state_id_code"] = df["state_id"].map(state_to_idx).astype(np.int64)

    # Run-Verzeichnis und Metadaten für Dokumentation.
    run_dir = ensure_run_dir()

    summary = {
        "seed": SEED,
        "num_seeds": int(NUM_SEEDS),
        "max_series": MAX_SERIES,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_scheduler": LR_SCHEDULER,
        "lr_factor": LR_FACTOR,
        "lr_patience": LR_PATIENCE,
        "hidden_size": HIDDEN_SIZE,
        "training_patience": PATIENCE,
        "feature_cols": feature_cols,
        "n_rows_total": int(len(df)),
        "n_series_total": int(df["series_id"].nunique()),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    save_json(run_dir / "run_config.json", summary)
    save_json(run_dir / "system_info.json", get_system_info())

    # MASE-Denominators werden ausschließlich auf Trainingsdaten berechnet.
    train_df = df[df["split"] == "train"].copy()
    mase_denoms = compute_mase_denominators(train_df, seasonality=7)

    seed_list = build_seed_list()

    all_seed_summaries = []
    best_overall_val_mase = float("inf")
    best_overall_seed = None
    best_overall_model_path = None

    for seed_value in seed_list:
        seed_run_dir = run_dir / f"seed_{seed_value}"
        seed_run_dir.mkdir(parents=True, exist_ok=True)

        seed_summary = train_one_seed(
            df=df,
            feature_cols=feature_cols,
            series_to_idx=series_to_idx,
            idx_to_series=idx_to_series,
            item_ids=item_ids,
            store_ids=store_ids,
            state_ids=state_ids,
            mase_denoms=mase_denoms,
            run_dir=seed_run_dir,
            seed_value=seed_value,
        )

        all_seed_summaries.append(seed_summary)

        if np.isfinite(seed_summary["best_val_mase"]) and seed_summary["best_val_mase"] < best_overall_val_mase:
            best_overall_val_mase = float(seed_summary["best_val_mase"])
            best_overall_seed = int(seed_value)
            best_overall_model_path = Path(seed_summary["best_model_path"])

    overall_summary = {
        "best_overall_seed": best_overall_seed,
        "best_overall_val_mase": float(best_overall_val_mase),
        "seed_summaries": all_seed_summaries,
    }
    save_json(run_dir / "overall_summary.json", overall_summary)

    if best_overall_model_path is not None and best_overall_model_path.exists():
        best_target_path = run_dir / f"best_model_overall_seed{best_overall_seed}.pt"
        best_target_path.write_bytes(best_overall_model_path.read_bytes())
        print("Saved best overall model:", best_target_path)

    print("Saved:", run_dir / "overall_summary.json")


if __name__ == "__main__":
    main()
