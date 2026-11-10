"""
Evaluation: window-level discrimination, calibration, alert/event-level metrics,
patient-level bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import config as C


def choose_threshold(y, p):
    """Threshold that maximises F1 on the validation set (chosen before looking at test)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def window_metrics(y, p, thr):
    yhat = p >= thr
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                prevalence=float(y.mean()), sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                f1=2 * ppv * sens / max(ppv + sens, 1e-12), brier=brier_score_loss(y, p),
                tp=tp, fp=fp, fn=fn, tn=tn, threshold=thr)


def bootstrap_ci(meta, y, p, n=C.N_BOOTSTRAP, seed=0):
    """95% CI for AUROC and AUPRC, resampling patients (clustered bootstrap)."""
    rng = np.random.default_rng(seed)
    pids = meta.patient_id.unique()
    groups = {pid: np.flatnonzero(meta.patient_id.to_numpy() == pid) for pid in pids}
    au, ap = [], []
    for _ in range(n):
        idx = np.concatenate([groups[g] for g in rng.choice(pids, len(pids), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        au.append(roc_auc_score(yy, p[idx])); ap.append(average_precision_score(yy, p[idx]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(au), q(ap)


def alerts_for_patient(risk, thr):
    """Debounced alerts: risk ≥ thr for ALERT_CONSECUTIVE_H hours, then refractory."""
    alerts = np.zeros(len(risk), dtype=int)
    run, last = 0, -10**9
    for t, r in enumerate(risk):
        run = run + 1 if (not np.isnan(r) and r >= thr) else 0
        if run >= C.ALERT_CONSECUTIVE_H and t - last >= C.ALERT_REFRACTORY_H:
            alerts[t] = 1
            last = t
    return alerts


def event_metrics(hourly_test, meta, p, thr):
    """Per-patient hourly risk timelines → alerts → event detection, lead time, false alerts."""
    risk_tl, rows_ev, alert_rows = {}, [], []
    total_alerts = false_alerts = 0
    monitored_h = 0
    meta = meta.assign(risk=p)
    for pid, pdf in hourly_test.groupby("patient_id", sort=False):
        H = len(pdf)
        risk = np.full(H, np.nan)
        m = meta[meta.patient_id == pid]
        risk[m.hour_index.to_numpy()] = m.risk.to_numpy()
        al = alerts_for_patient(risk, thr)
        risk_tl[pid] = (risk, al)
        onsets = np.flatnonzero(pdf.voc_onset.to_numpy())
        monitored_h += int((~np.isnan(risk)).sum())
        for t in np.flatnonzero(al):
            total_alerts += 1
            true = any((o > t) and (o - t <= C.HORIZON_H) for o in onsets)
            false_alerts += int(not true)
            alert_rows.append(dict(patient_id=pid, hour_index=int(t), true_alert=int(true)))
        for o in onsets:
            if o < C.CALIBRATION_HOURS + C.WINDOW_H:
                continue                                       # no usable look-back window
            win = np.arange(max(0, o - C.HORIZON_H), o)
            covered = (~np.isnan(risk[win])).mean()
            hits = win[al[win] == 1]
            rows_ev.append(dict(patient_id=pid, onset_hour_index=int(o), detected=int(len(hits) > 0),
                                lead_time_h=float(o - hits.min()) if len(hits) else np.nan,
                                max_risk_24h_before=float(np.nanmax(risk[win])) if covered > 0 else np.nan,
                                monitoring_coverage_24h=round(float(covered), 2)))
    ev = pd.DataFrame(rows_ev)
    weeks = monitored_h / 168
    summary = dict(
        n_events=len(ev), events_detected=int(ev.detected.sum()) if len(ev) else 0,
        event_sensitivity=float(ev.detected.mean()) if len(ev) else np.nan,
        median_lead_h=float(ev.lead_time_h.median()) if ev.detected.any() else np.nan,
        lead_q1_h=float(ev.lead_time_h.quantile(.25)) if ev.detected.any() else np.nan,
        lead_q3_h=float(ev.lead_time_h.quantile(.75)) if ev.detected.any() else np.nan,
        total_alerts=total_alerts, false_alerts=false_alerts,
        alert_ppv=(total_alerts - false_alerts) / max(total_alerts, 1),
        false_alerts_per_patient_week=false_alerts / max(weeks, 1e-9),
        monitored_patient_weeks=weeks)
    return summary, ev, risk_tl, pd.DataFrame(alert_rows)


def curve_points(y, p, kind="roc", n=150):
    if kind == "roc":
        a, b, _ = roc_curve(y, p)
        df = pd.DataFrame({"fpr": a, "tpr": b})
    else:
        prec, rec, _ = precision_recall_curve(y, p)
        df = pd.DataFrame({"recall": rec, "precision": prec}).iloc[::-1].reset_index(drop=True)
    if len(df) > n:
        idx = np.unique(np.r_[np.linspace(0, len(df) - 1, n).astype(int), len(df) - 1])
        df = df.iloc[idx]
    return df.round(4)


def calibration_table(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        s = b == k
        if s.any():
            rows.append(dict(bin=k + 1, mean_predicted=float(p[s].mean()), observed_rate=float(y[s].mean()), n=int(s.sum())))
    return pd.DataFrame(rows)
