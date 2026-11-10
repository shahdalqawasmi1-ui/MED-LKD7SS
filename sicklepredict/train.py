"""
End-to-end experiment: data → features → 5 models → evaluation → dashboard files.

Models compared (each maps to a claim in proposal §2.3):
  M1  Rule-based threshold (perfusion drop + HR rise, population scale)  – simple consumer-style alert
  M2  Random forest on summary features (population scale)               – prior-work style (Ji 2021, Vuong 2023)
  M3  Bi-LSTM + attention, population normalisation                       – tests "personalised baseline"
  M4  Bi-LSTM + attention, consumer PPG channels only, personal baseline  – tests "sensing specificity + fusion"
  M5  SicklePredict: Bi-LSTM + attention, all modalities, personal baseline (proposed)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C
from . import evaluate as E
from . import features as F
from . import simulate
from .bilstm import BiLSTMAttention, gradient_check

PROPOSED = "M5"
MODELS = {
    "M1": ("Rule-based threshold", "Perfusion-index drop + heart-rate rise vs. population norms; no learning"),
    "M2": ("Random forest, summary features", "Prior-work style: engineered 24-h summary features, population scale"),
    "M3": ("Bi-LSTM + attention, population baseline", "All modalities, but one population-wide normalisation"),
    "M4": ("Bi-LSTM + attention, consumer sensors", "Personal baseline, but only consumer-watch channels (520/660/940 PPG, HR, steps)"),
    "M5": ("SicklePredict (proposed)", "Bi-LSTM + attention, all modalities, personal circadian baseline"),
}


def load_data():
    if not (C.DATA_DIR / "wearable_hourly.csv.gz").exists():
        print("No data found – generating the synthetic cohort …")
        simulate.main()
    patients = pd.read_csv(C.DATA_DIR / "patients.csv")
    events = pd.read_csv(C.DATA_DIR / "voc_events.csv", parse_dates=["onset_time"])
    hourly = pd.read_csv(C.DATA_DIR / "wearable_hourly.csv.gz", parse_dates=["timestamp"])
    return patients, hourly, events


def summary_features(X):
    """Window summaries for M2: per channel mean(24h), mean(last 6h), trend, std, min, max."""
    n_sig = len(C.SIGNAL_CHANNELS)
    S = X[:, :, :n_sig]
    feats = [S.mean(1), S[:, -6:].mean(1), S[:, -6:].mean(1) - S[:, :6].mean(1),
             S.std(1), S.min(1), S.max(1), X[:, :, n_sig].mean(1, keepdims=True)]
    return np.concatenate(feats, axis=1)


def rule_score(X):
    ch = C.SIGNAL_CHANNELS
    pi = X[:, -6:, ch.index("ppg_pi_520")].mean(1)
    hr = X[:, -6:, ch.index("heart_rate_bpm")].mean(1)
    return -pi + 0.5 * hr


def platt(score_val, y_val, score_test):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(score_val[:, None], y_val)
    return lr.predict_proba(score_val[:, None])[:, 1], lr.predict_proba(score_test[:, None])[:, 1]


def permutation_importance(model, X, y, cols, base_auc, groups, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, chans in groups.items():
        idx = [cols.index(c) for c in chans]
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, :, idx] = X[perm][:, :, idx]
        auc = roc_auc_score(y, model.predict_logit(Xp))
        rows.append(dict(feature=name, auroc_drop=base_auc - auc))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.DASH_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6  Checking Bi-LSTM gradients …", end=" ")
    print(f"max relative error {gradient_check():.1e}")

    print("2/6  Loading data …")
    patients, hourly, events = load_data()
    ids = {s: patients.loc[patients.data_split == s, "patient_id"].tolist() for s in ("train", "validation", "test")}

    print("3/6  Building personal and population baselines, 24-h windows …")
    ch = C.SIGNAL_CHANNELS
    feat_pers, baselines = F.normalise(hourly, ids["train"], ch, "personal")
    feat_pop, _ = F.normalise(hourly, ids["train"], ch, "population")
    W = {}
    for split in ("train", "validation", "test"):
        Xp, y, meta = F.make_windows(hourly, feat_pers, ids[split])
        Xq, y2, _ = F.make_windows(hourly, feat_pop, ids[split])
        assert np.array_equal(y, y2)
        W[split] = dict(pers=Xp, pop=Xq, y=y, meta=meta)
        print(f"     {split:<10} windows {len(y):>7,}   positive {int(y.sum()):>5,} ({100 * y.mean():.2f} %)")

    rng = np.random.default_rng(C.SEED)
    ytr = W["train"]["y"]
    pos = np.flatnonzero(ytr == 1)
    neg = rng.choice(np.flatnonzero(ytr == 0), min(len(pos) * C.NEG_PER_POS_TRAIN, int((ytr == 0).sum())), replace=False)
    tr_idx = rng.permutation(np.r_[pos, neg])
    print(f"     training sample: {len(pos):,} positive + {len(neg):,} negative windows")

    yva, yte = W["validation"]["y"], W["test"]["y"]
    meta_te = W["test"]["meta"]
    hourly_te = hourly[hourly.patient_id.isin(ids["test"])].sort_values(["patient_id", "hour_index"])
    results, curves_roc, curves_pr, calib, histories = [], [], [], [], []
    store = {}

    print("4/6  Training and evaluating models …")
    for key, (name, desc) in MODELS.items():
        print(f"  {key}  {name}")
        if key == "M1":
            s_va, s_te = rule_score(W["validation"]["pop"]), rule_score(W["test"]["pop"])
            p_va, p_te = platt(s_va, yva, s_te)
            model = None
        elif key == "M2":
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED)
            rf.fit(summary_features(W["train"]["pop"][tr_idx]), ytr[tr_idx])
            s_va = rf.predict_proba(summary_features(W["validation"]["pop"]))[:, 1]
            s_te = rf.predict_proba(summary_features(W["test"]["pop"]))[:, 1]
            p_va, p_te = platt(s_va, yva, s_te)
            model = rf
        else:
            src = "pop" if key == "M3" else "pers"
            cols = F.channel_index(C.CONSUMER_PPG_CHANNELS if key == "M4" else ch)
            Xtr = W["train"][src][:, :, cols]
            Xva = W["validation"][src][:, :, cols]
            Xte = W["test"][src][:, :, cols]
            model = BiLSTMAttention(len(cols), C.HIDDEN, C.ATTN_DIM, C.DENSE, seed=C.SEED)
            model.fit(Xtr, ytr, Xva, yva, epochs=C.EPOCHS, batch=C.BATCH, lr=C.LR,
                      weight_decay=C.WEIGHT_DECAY, dropout=C.DROPOUT, clip=C.GRAD_CLIP,
                      patience=C.PATIENCE, seed=C.SEED, verbose=True,
                      neg_ratio=C.NEG_PER_POS_TRAIN, input_noise=C.INPUT_NOISE)
            model.calibrate(model.predict_logit(Xva), yva)
            p_va = model.predict_proba(Xva)
            p_te, att = model.predict_proba(Xte, return_attention=True)
            model.save(C.MODEL_DIR / f"{key}_bilstm.npz")
            for h in model.history:
                histories.append(dict(model=key, **h))
            store[key] = dict(model=model, cols=cols, Xte=Xte, att=att)

        thr = E.choose_threshold(yva, p_va)
        wm = E.window_metrics(yte, p_te, thr)
        (a_lo, a_hi), (p_lo, p_hi) = E.bootstrap_ci(meta_te, yte, p_te)
        evs, ev_tab, tl, alerts = E.event_metrics(hourly_te, meta_te, p_te, thr)
        results.append(dict(model_id=key, model=name, description=desc, **wm,
                            auroc_ci_low=a_lo, auroc_ci_high=a_hi, auprc_ci_low=p_lo, auprc_ci_high=p_hi, **evs))
        cr = E.curve_points(yte, p_te, "roc"); cr.insert(0, "model_id", key); curves_roc.append(cr)
        cp = E.curve_points(yte, p_te, "pr"); cp.insert(0, "model_id", key); curves_pr.append(cp)
        cb = E.calibration_table(yte, p_te); cb.insert(0, "model_id", key); calib.append(cb)
        store.setdefault(key, {}).update(p_te=p_te, thr=thr, ev=ev_tab, tl=tl, alerts=alerts)
        print(f"     test AUROC {wm['auroc']:.3f} ({a_lo:.3f}–{a_hi:.3f})  AUPRC {wm['auprc']:.3f}  "
              f"event sens {evs['event_sensitivity']:.2f}  median lead {evs['median_lead_h']:.0f} h  "
              f"false alerts/pt-week {evs['false_alerts_per_patient_week']:.2f}")

    print("5/6  Explaining the proposed model …")
    P = store[PROPOSED]
    model, Xte, p_te, thr = P["model"], P["Xte"], P["p_te"], P["thr"]
    base_auc = roc_auc_score(yte, model.predict_logit(Xte))
    cols_named = ch + C.EXTRA_INPUTS
    imp_grp = permutation_importance(model, Xte, yte, cols_named, base_auc, C.MODALITY_GROUPS)
    imp_grp.insert(0, "level", "modality")
    imp_ch = permutation_importance(model, Xte, yte, cols_named, base_auc, {c: [c] for c in ch})
    imp_ch.insert(0, "level", "channel")
    importance = pd.concat([imp_grp, imp_ch], ignore_index=True)
    importance["auroc_drop"] = importance.auroc_drop.round(4)

    att = P["att"]
    offs = np.arange(-C.WINDOW_H + 1, 1)
    attention = pd.DataFrame({"hours_before_prediction": offs,       # -23 … 0 (0 = latest hour)
                              "attention_pre_crisis": att[yte == 1].mean(0),
                              "attention_no_crisis": att[yte == 0].mean(0)}).round(5)

    # subgroup performance (proposal §10: fairness across age, sex, genotype)
    pt = patients.set_index("patient_id")
    m = meta_te.assign(y=yte, p=p_te)
    m["sex"] = m.patient_id.map(pt.sex)
    m["genotype_group"] = m.patient_id.map(pt.genotype).map(
        {"HbSS": "HbSS / HbSβ0", "HbSβ0": "HbSS / HbSβ0", "HbSβ+": "HbSC / HbSβ+", "HbSC": "HbSC / HbSβ+"})
    m["age_group"] = pd.cut(m.patient_id.map(pt.age_years), [0, 17, 30, 200], labels=["< 18", "18–30", "> 30"]).astype(str)
    m["hydroxyurea"] = m.patient_id.map(pt.hydroxyurea).map({1: "On hydroxyurea", 0: "No hydroxyurea"})
    ev5 = P["ev"].copy()
    for col in ("sex", "genotype_group", "age_group", "hydroxyurea"):
        ev5[col] = ev5.patient_id.map(m.drop_duplicates("patient_id").set_index("patient_id")[col])
    sub_rows = []
    for col, label in (("sex", "Sex"), ("genotype_group", "Genotype"), ("age_group", "Age"), ("hydroxyurea", "Therapy")):
        for lvl, g in m.groupby(col):
            e = ev5[ev5[col] == lvl]
            sub_rows.append(dict(dimension=label, subgroup=lvl, patients=g.patient_id.nunique(),
                                 windows=len(g), events=len(e),
                                 auroc=roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan,
                                 event_sensitivity=e.detected.mean() if len(e) else np.nan))
    subgroups = pd.DataFrame(sub_rows).round(3)

    print("6/6  Writing dashboard files …")
    D = C.DASH_DIR
    res = pd.DataFrame(results)
    res.round(4).to_csv(D / "model_comparison.csv", index=False)
    pd.concat(curves_roc).to_csv(D / "roc_curves.csv", index=False)
    pd.concat(curves_pr).to_csv(D / "pr_curves.csv", index=False)
    pd.concat(calib).round(4).to_csv(D / "calibration.csv", index=False)
    pd.DataFrame(histories).round(4).to_csv(D / "training_history.csv", index=False)
    importance.to_csv(D / "feature_importance.csv", index=False)
    attention.to_csv(D / "attention_profile.csv", index=False)
    subgroups.to_csv(D / "subgroup_performance.csv", index=False)

    # test events with clinical context
    ev = ev5.merge(events[["patient_id", "onset_hour_index", "event_id", "onset_time", "care_setting",
                           "likely_trigger", "peak_pain_score"]], on=["patient_id", "onset_hour_index"], how="left")
    ev["genotype"] = ev.patient_id.map(pt.genotype)
    ev = ev[["event_id", "patient_id", "onset_time", "genotype", "care_setting", "likely_trigger",
             "peak_pain_score", "detected", "lead_time_h", "max_risk_24h_before", "monitoring_coverage_24h"]]
    ev.round({"lead_time_h": 1, "max_risk_24h_before": 3}).to_csv(D / "test_events.csv", index=False)

    # per-patient timelines for the monitor view (proposed model)
    tl_rows = []
    for pid, pdf in hourly_te.groupby("patient_id", sort=False):
        risk, al = P["tl"][pid]
        tl_rows.append(pd.DataFrame({
            "patient_id": pid, "timestamp": pdf.timestamp.dt.strftime("%Y-%m-%d %H:00").to_numpy(),
            "hour_index": pdf.hour_index.to_numpy(), "risk": np.round(risk, 4), "alert": al,
            "voc_active": pdf.voc_active.to_numpy(), "voc_onset": pdf.voc_onset.to_numpy(),
            "ppg_pi_810": pdf.ppg_pi_810.to_numpy(), "skin_temp_c": pdf.skin_temp_c.to_numpy(),
            "eda_tonic_us": pdf.eda_tonic_us.to_numpy(), "heart_rate_bpm": pdf.heart_rate_bpm.to_numpy(),
            "spo2_est": pdf.spo2_est.to_numpy(), "device_worn": pdf.device_worn.to_numpy()}))
    timeline = pd.concat(tl_rows, ignore_index=True)
    timeline.to_csv(D / "patient_timeline.csv", index=False)

    # test-patient roster
    al = P["alerts"]
    roster = patients[patients.data_split == "test"].drop(columns=["data_split", "sim_signature_strength"]).copy()
    g_ev = ev.groupby("patient_id")
    roster["events_evaluable"] = roster.patient_id.map(g_ev.size()).fillna(0).astype(int)
    roster["events_detected"] = roster.patient_id.map(g_ev.detected.sum()).fillna(0).astype(int)
    roster["alerts_total"] = roster.patient_id.map(al.groupby("patient_id").size()).fillna(0).astype(int)
    roster["alerts_false"] = roster.patient_id.map(al[al.true_alert == 0].groupby("patient_id").size()).fillna(0).astype(int)
    last = timeline.dropna(subset=["risk"]).groupby("patient_id").risk.last()
    roster["latest_risk"] = roster.patient_id.map(last).round(3)
    roster["peak_risk"] = roster.patient_id.map(timeline.groupby("patient_id").risk.max()).round(3)
    roster.to_csv(D / "test_patients.csv", index=False)

    n_events_total = len(events)
    summary = pd.DataFrame([
        ("patients_enrolled", len(patients)), ("monitoring_days_per_patient", C.MONITOR_DAYS),
        ("patient_hours_recorded", len(hourly)), ("wear_time_percent", round(100 * hourly.device_worn.mean(), 1)),
        ("voc_events_total", n_events_total),
        ("patients_train", len(ids["train"])), ("patients_validation", len(ids["validation"])),
        ("patients_test", len(ids["test"])),
        ("test_windows", len(yte)), ("test_positive_windows", int(yte.sum())),
        ("test_prevalence_percent", round(100 * yte.mean(), 2)),
        ("alert_threshold_proposed", round(thr, 4)),
        ("prediction_horizon_h", C.HORIZON_H), ("lookback_window_h", C.WINDOW_H),
        ("calibration_days", C.CALIBRATION_HOURS // 24), ("recovery_exclusion_h", C.RECOVERY_H),
        ("alert_consecutive_h", C.ALERT_CONSECUTIVE_H), ("alert_refractory_h", C.ALERT_REFRACTORY_H),
        ("runtime_minutes", round((time.time() - t0) / 60, 1)),
    ], columns=["metric", "value"])
    summary.to_csv(D / "cohort_summary.csv", index=False)

    # personal baselines for the test patients (needed by predict.py on new data)
    with open(C.MODEL_DIR / "run_info.json", "w") as f:
        pop_med, pop_scl = F.population_stats(hourly, ids["train"], ch)
        json.dump({"proposed_model": f"{PROPOSED}_bilstm.npz", "threshold": thr,
                   "channels": ch, "population_scale": [float(v) for v in pop_scl],
                   "results": res[["model_id", "model", "auroc", "auprc", "event_sensitivity",
                                                     "median_lead_h", "false_alerts_per_patient_week"]].round(4).to_dict("records")},
                  f, indent=2, ensure_ascii=False)

    print("\nRESULTS (held-out test patients)")
    print(res[["model_id", "model", "auroc", "auprc", "sensitivity", "specificity", "event_sensitivity",
               "median_lead_h", "false_alerts_per_patient_week"]].round(3).to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Dashboard files: {D}")


if __name__ == "__main__":
    main()
