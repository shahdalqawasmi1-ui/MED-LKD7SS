"""
Score new wristband data with the trained SicklePredict model.

Usage:
    python -m sicklepredict.predict  path/to/new_wearable_hourly.csv  [output.csv]

The input CSV needs the columns of `wearable_hourly` (see data/data_dictionary.csv):
patient_id, timestamp, the 15 signal channels and device_worn. Each patient needs at least
7 days of data: the first 7 days become that patient's personal baseline (calibration week)
and are not scored.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .bilstm import BiLSTMAttention
from .evaluate import alerts_for_patient


def risk_level(p, thr):
    return np.where(np.isnan(p), "no data", np.where(p >= thr, "HIGH", np.where(p >= thr / 2, "MODERATE", "LOW")))


def score(df: pd.DataFrame) -> pd.DataFrame:
    info = json.load(open(C.MODEL_DIR / "run_info.json"))
    model = BiLSTMAttention.load(C.MODEL_DIR / info["proposed_model"])
    thr = info["threshold"]
    ch = info["channels"]
    pop_scl = np.array(info["population_scale"])

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "voc_active" not in df:
        df["voc_active"] = 0
    out = []
    for pid, pdf in df.groupby("patient_id", sort=False):
        pdf = pdf.sort_values("timestamp").reset_index(drop=True)
        pdf["hour_index"] = np.arange(len(pdf))
        if len(pdf) < C.CALIBRATION_HOURS + C.WINDOW_H:
            print(f"  {pid}: fewer than 8 days of data – skipped")
            continue
        X = F.transform(pdf, ch)
        blk = pdf.timestamp.dt.hour.to_numpy() // 6
        med, scl = F.personal_baseline(pdf, ch, pop_scl)
        Z = (X - med[blk]) / scl[blk]
        Z[:, ch.index("ambient_temp_c")] = (pdf["ambient_temp_c"].to_numpy() - F.AMBIENT_CENTER) / F.AMBIENT_SCALE
        Z = np.clip(np.nan_to_num(pd.DataFrame(Z).ffill(limit=2).to_numpy(), nan=0.0), -6, 6)
        hod = pdf.timestamp.dt.hour.to_numpy()
        feats = np.c_[Z, pdf.device_worn.to_numpy(), np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)].astype(np.float32)

        W = C.WINDOW_H
        idx = np.arange(max(C.CALIBRATION_HOURS, W - 1), len(pdf))
        worn = pdf.device_worn.to_numpy()
        cs = np.r_[0, np.cumsum(worn)]
        idx = idx[(cs[idx + 1] - cs[idx + 1 - W]) / W >= 0.5]
        win = np.lib.stride_tricks.sliding_window_view(feats, (W, feats.shape[1]))[:, 0]
        risk = np.full(len(pdf), np.nan)
        if len(idx):
            p, att = model.predict_proba(win[idx - (W - 1)], return_attention=True)
            risk[idx] = p
        al = alerts_for_patient(risk, thr)
        out.append(pd.DataFrame({"patient_id": pid, "timestamp": pdf.timestamp,
                                 "risk_score": np.round(risk, 4), "risk_level": risk_level(risk, thr),
                                 "alert": al}))
    return pd.concat(out, ignore_index=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "risk_scores.csv"
    res = score(pd.read_csv(src))
    res.to_csv(dst, index=False)
    print(f"Scored {res.patient_id.nunique()} patient(s), {res.alert.sum()} alert(s). Saved to {dst}")


if __name__ == "__main__":
    main()
