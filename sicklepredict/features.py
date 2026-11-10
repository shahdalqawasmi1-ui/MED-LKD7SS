"""
Feature pipeline: personal-baseline normalisation and sliding windows.

Personal baseline (proposal §4.1): for every patient and channel, the median and robust
spread are learned from that patient's own calibration week, separately for four
6-hour blocks of the day (circadian baseline). Every later hour is then expressed as
"how many personal standard deviations away from my usual value at this time of day".

Population normalisation (ablation): one median/spread per channel computed from all
training patients, i.e. a single population-wide threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LOG_CHANNELS = {"ppg_pi_520", "ppg_pi_660", "ppg_pi_940", "ppg_pi_810", "eda_tonic_us"}
LOG1P_CHANNELS = {"steps_per_hour"}
CONTEXT_CHANNELS = {"ambient_temp_c"}          # environment, not physiology: never personalised
AMBIENT_CENTER, AMBIENT_SCALE = 24.0, 8.0


def transform(df: pd.DataFrame, channels) -> np.ndarray:
    X = df[channels].to_numpy(dtype=float).copy()
    for j, c in enumerate(channels):
        if c in LOG_CHANNELS:
            X[:, j] = np.log(np.clip(X[:, j], 1e-3, None))
        elif c in LOG1P_CHANNELS:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return X


def _robust(x):
    x = x[~np.isnan(x)]
    if len(x) < 6:
        return np.nan, np.nan
    med = np.median(x)
    s = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349
    return med, s


def population_stats(hourly: pd.DataFrame, train_ids, channels):
    sub = hourly[hourly.patient_id.isin(train_ids) & (hourly.eval_exclude == 0)]
    X = transform(sub, channels)
    med = np.nanmedian(X, axis=0)
    s = (np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)) / 1.349
    return med, np.where(s > 1e-6, s, 1.0)


def personal_baseline(pdf: pd.DataFrame, channels, pop_scale):
    """Circadian personal baseline from the calibration week -> (4, D) medians and scales."""
    calib = pdf[(pdf.hour_index < C.CALIBRATION_HOURS) & (pdf.voc_active == 0)]
    X = transform(calib, channels)
    blk = (calib.timestamp.dt.hour.to_numpy() // 6)
    med = np.zeros((4, len(channels)))
    scl = np.zeros((4, len(channels)))
    for j in range(len(channels)):
        m_all, s_all = _robust(X[:, j])
        for b in range(4):
            m, s = _robust(X[blk == b, j])
            med[b, j] = m if not np.isnan(m) else m_all
            scl[b, j] = s if not np.isnan(s) else s_all
        # never let a personal spread collapse below 25 % of the population spread
        scl[:, j] = np.maximum(np.nan_to_num(scl[:, j], nan=pop_scale[j]), 0.25 * pop_scale[j])
        med[:, j] = np.nan_to_num(med[:, j], nan=0.0)
    return med, scl


def normalise(hourly: pd.DataFrame, train_ids, channels, mode="personal"):
    """Returns dict patient_id -> (H, D_in) float32 matrix of model inputs."""
    pop_med, pop_scl = population_stats(hourly, train_ids, channels)
    out, baselines = {}, {}
    for pid, pdf in hourly.groupby("patient_id", sort=False):
        pdf = pdf.sort_values("hour_index")
        X = transform(pdf, channels)
        blk = pdf.timestamp.dt.hour.to_numpy() // 6
        if mode == "personal":
            med, scl = personal_baseline(pdf, channels, pop_scl)
            Z = (X - med[blk]) / scl[blk]
            baselines[pid] = (med, scl)
        else:
            Z = (X - pop_med) / pop_scl
        for j, c in enumerate(channels):
            if c in CONTEXT_CHANNELS:
                Z[:, j] = (pdf[c].to_numpy() - AMBIENT_CENTER) / AMBIENT_SCALE
        Z = pd.DataFrame(Z).ffill(limit=2).to_numpy()   # carry short gaps forward ≤ 2 h
        Z = np.clip(np.nan_to_num(Z, nan=0.0), -6, 6)
        hod = pdf.timestamp.dt.hour.to_numpy()
        extra = np.c_[pdf.device_worn.to_numpy(), np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)]
        out[pid] = np.c_[Z, extra].astype(np.float32)
    return out, baselines


def make_windows(hourly: pd.DataFrame, feats: dict, patient_ids, min_worn=0.5):
    """Sliding 24-h windows ending at every eligible hour. Returns X, y, meta."""
    Xs, ys, metas = [], [], []
    W = C.WINDOW_H
    for pid in patient_ids:
        pdf = hourly[hourly.patient_id == pid].sort_values("hour_index")
        F = feats[pid]
        worn = pdf.device_worn.to_numpy()
        excl = pdf.eval_exclude.to_numpy()
        lab = pdf.label_voc_next_24h.to_numpy()
        idx = np.arange(W - 1, len(pdf))
        cs = np.r_[0, np.cumsum(worn)]
        worn_frac = (cs[idx + 1] - cs[idx + 1 - W]) / W
        idx = idx[(excl[idx] == 0) & (worn_frac >= min_worn)]
        if len(idx) == 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(F, (W, F.shape[1]))[:, 0]  # (H-W+1, W, D)
        Xs.append(win[idx - (W - 1)])
        ys.append(lab[idx])
        metas.append(pd.DataFrame({
            "patient_id": pid, "hour_index": idx,
            "timestamp": pdf.timestamp.to_numpy()[idx],
            "hours_to_next_onset": pdf.hours_to_next_onset.to_numpy()[idx]}))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    return X, y, pd.concat(metas, ignore_index=True)


def channel_index(channels_used, all_channels=None):
    """Column indices (in the full input matrix) for a subset of channels + extras."""
    all_channels = all_channels or C.SIGNAL_CHANNELS
    idx = [all_channels.index(c) for c in channels_used]
    n = len(all_channels)
    return idx + [n, n + 1, n + 2]
