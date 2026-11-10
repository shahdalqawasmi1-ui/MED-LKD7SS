"""
Synthetic SCD cohort + wearable data generator for the SicklePredict prototype.

What it produces (one row per patient-hour):
  * the 15 wearable/context channels described in the proposal (Section 3),
  * VOC events with an ASSUMED pre-crisis physiological signature,
  * realistic nuisance: circadian rhythm, activity, motion artefacts, non-wear,
    cold exposure, emotional stress, infections, and "aborted" vasoconstriction
    episodes that look like a prodrome but never become a crisis.

IMPORTANT: this is simulated data. Every physiological effect below is an assumption
written into the code, so any model performance obtained on it demonstrates that the
pipeline works — it is NOT evidence that the device predicts VOC in real patients.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config as C

# Mean monthly outdoor temperature, central Saudi Arabia (approx., °C), Jan..Dec
MONTHLY_TEMP = np.array([13, 16, 21, 26, 32, 35, 36, 36, 33, 27, 20, 15], dtype=float)

GENOTYPES = ["HbSS", "HbSβ0", "HbSβ+", "HbSC"]
GENOTYPE_P = [0.68, 0.10, 0.10, 0.12]
SEVERITY = {"HbSS": 1.15, "HbSβ0": 1.15, "HbSβ+": 0.75, "HbSC": 0.70}
REGIONS = ["Eastern", "Southwestern", "Central", "Western"]
REGION_P = [0.45, 0.30, 0.15, 0.10]


def _ar1(rng, n, phi, sd):
    """AR(1) noise with stationary standard deviation `sd`."""
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _episodes(rng, n_hours, rate_per_day, min_h, max_h):
    """Random on/off episodes; returns a 0/1 mask and list of (start, length)."""
    mask = np.zeros(n_hours)
    eps = []
    n = rng.poisson(rate_per_day * n_hours / 24)
    for _ in range(n):
        s = rng.integers(0, n_hours)
        L = rng.integers(min_h, max_h + 1)
        mask[s:s + L] = 1
        eps.append((s, L))
    return mask, eps


def _rolling_max_past(x, w):
    """max over the previous w hours (inclusive)."""
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1):i + 1].max()
    return out


def make_patient(idx: int, rng: np.random.Generator):
    H = C.HOURS
    pid = f"SP-{idx:03d}"

    # ------------------------------------------------------------------ profile
    sex = rng.choice(["F", "M"])
    age = int(np.clip(rng.normal(26, 9), 14, 60))
    genotype = rng.choice(GENOTYPES, p=GENOTYPE_P)
    region = rng.choice(REGIONS, p=REGION_P)
    hu = rng.random() < (0.55 if genotype in ("HbSS", "HbSβ0") else 0.20)
    hbf_base = {"Eastern": 17, "Southwestern": 8, "Central": 11, "Western": 10}[region]
    hbf = float(np.clip(rng.normal(hbf_base + (5 if hu else 0), 4), 1, 35))
    hb = {"HbSS": 8.2, "HbSβ0": 8.0, "HbSβ+": 10.0, "HbSC": 11.0}[genotype]
    hb = float(np.round(np.clip(rng.normal(hb, 0.9), 6.0, 13.5), 1))
    bmi = float(np.round(np.clip(rng.normal(21.5, 3.5), 15, 35), 1))
    rate = C.MEAN_VOC_PER_YEAR * rng.lognormal(0, 0.40) * SEVERITY[genotype] \
        * (0.80 if hu else 1.0) * np.exp(-0.02 * (hbf - 12))
    rate = float(np.clip(rate, 3.0, 24.0))
    prior_ed = int(rng.poisson(rate * 0.8))
    prior_adm = int(rng.binomial(prior_ed, 0.25))

    start = datetime(2026, 1, 1) + timedelta(days=int(rng.integers(0, 240)))
    ts = pd.date_range(start, periods=H, freq="h")
    hod = ts.hour.values
    month = ts.month.values - 1

    # Personal physiology (these baseline differences are why a personal baseline matters)
    base = dict(
        pi520=rng.lognormal(np.log(1.1), 0.30), pi660=rng.lognormal(np.log(1.9), 0.30),
        pi940=rng.lognormal(np.log(2.6), 0.28), pi810=rng.lognormal(np.log(2.2), 0.30),
        spo2=float(np.clip(rng.normal(95.0 if genotype in ("HbSS", "HbSβ0") else 96.5, 1.4), 89, 99.5)),
        transit=rng.normal(215, 20), hr=rng.normal(84 + (8.2 - hb) * 1.5, 7),
        rmssd=float(np.clip(rng.normal(32, 9), 12, 70)), eda=rng.lognormal(np.log(2.0), 0.55),
        scr=rng.uniform(1.8, 3.5), skin=rng.normal(32.8, 0.6), steps=rng.uniform(250, 750),
    )
    # How strongly / through which pathway this patient expresses a prodrome
    strength = float(rng.beta(3.0, 1.6))
    w = dict(vaso=rng.uniform(0.3, 1.2), oxy=rng.uniform(0.3, 1.2),
             rheo=rng.uniform(0.3, 1.2), auto=rng.uniform(0.3, 1.2))

    # ------------------------------------------------------------------ context
    bed = rng.uniform(22, 25.5)
    sleep = np.zeros(H)
    for d in range(C.MONITOR_DAYS + 1):
        b = 24 * d + bed + rng.normal(0, 0.7) - 0  # hours from start
        e = b + rng.uniform(6.5, 8.5)
        lo, hi = int(max(0, round(b))), int(min(H, round(e)))
        if lo < H:
            sleep[lo:hi] = 1
    awake = 1 - sleep

    amb = MONTHLY_TEMP[month] + rng.normal(0, 2) \
        + 6 * np.sin((hod - 9) / 24 * 2 * np.pi) + _ar1(rng, H, 0.95, 1.0)
    amb = np.clip(amb, 0, 50)

    day_activity = np.repeat(rng.lognormal(0, 0.35, C.MONITOR_DAYS + 1), 24)[:H]
    steps = awake * rng.gamma(1.3, base["steps"] / 1.3, H) * day_activity \
        + sleep * rng.poisson(3, H)
    steps = np.minimum(steps, 6000)

    cold, _ = _episodes(rng, H, 0.30, 1, 4)           # AC / cold water / cold night exposure
    stress, _ = _episodes(rng, H, 0.9, 1, 3)          # emotional / exam stress
    stress *= awake
    infection = np.zeros(H)
    if rng.random() < 0.30:
        s = rng.integers(48, H - 120)
        infection[s:s + rng.integers(72, 121)] = 1
    heat = ((amb > 40) & (steps > base["steps"] * 1.2)).astype(float)

    # ------------------------------------------------------------------ VOC events
    mult = 1 + 1.8 * _rolling_max_past(cold, 24) + 0.6 * _rolling_max_past(heat, 24) + 3.0 * infection
    lam = rate / 8760 / mult.mean() * mult
    onsets, crises = [], []
    t = 48
    while t < H:
        if rng.random() < lam[t]:
            dur = int(rng.integers(C.CRISIS_MIN_DAYS * 24, C.CRISIS_MAX_DAYS * 24 + 1))
            lead = int(rng.integers(C.PRODROME_MIN_H, C.PRODROME_MAX_H + 1))
            onsets.append(t)
            crises.append((t, dur, lead))
            t += dur + C.REFRACTORY_DAYS * 24
        else:
            t += 1

    prodrome = np.zeros(H)
    crisis = np.zeros(H)
    recovery = np.zeros(H)
    events = []
    for k, (t0, dur, lead) in enumerate(crises):
        e = strength * rng.uniform(0.6, 1.2)
        idx = np.arange(max(0, t0 - lead), t0)
        prodrome[idx] = np.maximum(prodrome[idx], e * ((idx - (t0 - lead) + 1) / lead) ** 1.0)
        crisis[t0:t0 + dur] = 1
        rec = np.arange(t0 + dur, min(H, t0 + dur + C.RECOVERY_H))
        recovery[rec] = 1 - (rec - (t0 + dur)) / C.RECOVERY_H
        trig = "infection" if infection[t0] else ("cold exposure" if cold[max(0, t0 - 24):t0].any()
                                                   else ("heat / dehydration" if heat[max(0, t0 - 24):t0].any() else "none identified"))
        setting = rng.choice(["home-managed", "ED visit", "admitted"], p=[0.30, 0.45, 0.25])
        events.append(dict(
            event_id=f"{pid}-E{k + 1}", patient_id=pid, onset_time=ts[t0], onset_hour_index=t0,
            crisis_duration_h=min(dur, H - t0), care_setting=setting,
            likely_trigger=trig, peak_pain_score=int(rng.integers(6, 11)),
            sim_prodrome_lead_h=lead, sim_prodrome_intensity=round(e, 3)))

    # Aborted vasoconstriction episodes: prodrome-like, partial signature, no crisis
    aborted = np.zeros(H)
    for _ in range(rng.poisson(H / 24 / 10)):
        s, L = int(rng.integers(0, H)), int(rng.integers(4, 15))
        a = np.arange(s, min(H, s + L))
        tri = 1 - np.abs((a - s) / max(L - 1, 1) * 2 - 1)
        aborted[a] = np.maximum(aborted[a], rng.uniform(0.4, 0.9) * tri)

    in_crisis = np.clip(crisis + recovery, 0, 1)
    p, ab = prodrome, aborted
    vaso = w["vaso"] * p + 0.8 * ab + 0.9 * cold + 0.6 * in_crisis

    # ------------------------------------------------------------------ signals
    steps = steps * (1 - 0.35 * np.clip(p, 0, 1)) * (1 - 0.8 * crisis) * (1 - 0.5 * infection)
    def pi(b, k):
        return b * np.exp(0.22 * sleep + 0.02 * (amb - 24) * 0.3 - k * vaso
                          + _ar1(rng, H, 0.7, 0.08)) * (1 + 0.15 * infection)

    pi520 = pi(base["pi520"], 0.38)
    pi660 = pi(base["pi660"], 0.33)
    pi940 = pi(base["pi940"], 0.30)
    pi810 = pi(base["pi810"], 0.45)

    ror = (110 - base["spo2"]) / 25 + 0.06 * w["oxy"] * p + 0.04 * in_crisis \
        + 0.015 * infection + _ar1(rng, H, 0.6, 0.013)
    spo2 = np.clip(110 - 25 * ror, 80, 100)

    hr = base["hr"] - 10 * sleep + 0.008 * steps + 8 * w["auto"] * p + 4 * ab \
        + 6 * stress + 15 * in_crisis + 12 * infection + _ar1(rng, H, 0.6, 3.0)
    transit = base["transit"] + 32 * w["rheo"] * p + 12 * in_crisis + 5 * sleep \
        - 0.4 * (hr - base["hr"]) + _ar1(rng, H, 0.6, 6.0)
    rmssd = base["rmssd"] * (1 + 0.3 * sleep) * np.exp(-0.00012 * steps) \
        * (1 - 0.32 * w["auto"] * p - 0.15 * stress - 0.3 * in_crisis - 0.25 * infection) \
        * np.exp(_ar1(rng, H, 0.5, 0.12))
    rmssd = np.clip(rmssd, 5, None)
    eda = base["eda"] * (1 - 0.3 * sleep) * (1 + 0.65 * w["auto"] * p + 0.3 * ab + 0.55 * stress
                                             + 0.5 * in_crisis) * np.exp(_ar1(rng, H, 0.7, 0.12))
    scr = np.clip(base["scr"] * (awake + 0.25 * sleep) + 1.6 * w["auto"] * p + 1.2 * stress
                  + 0.8 * in_crisis + rng.normal(0, 0.45, H), 0, None)
    skin = base["skin"] + 1.0 * sleep + 0.07 * (amb - 24) * 0.4 - 0.00035 * steps \
        - 1.1 * w["vaso"] * p - 0.6 * ab - 1.4 * cold - 0.6 * in_crisis + 0.9 * infection \
        + _ar1(rng, H, 0.75, 0.18)

    # Motion artefacts and non-wear
    quality = np.clip(1 - 0.00030 * steps - np.abs(rng.normal(0, 0.07, H)), 0, 1)
    bad = quality < 0.45
    for arr, sd in ((pi520, 0.5), (pi660, 0.5), (pi940, 0.5), (pi810, 0.5)):
        arr[bad] *= np.exp(rng.normal(0, sd, bad.sum()))
    ror[bad] += rng.normal(0, 0.06, bad.sum())
    spo2 = np.clip(110 - 25 * ror, 80, 100)

    worn = np.ones(H)
    for d in range(C.MONITOR_DAYS):           # daily charging
        if rng.random() < 0.85:
            s = 24 * d + int(rng.integers(7, 22))
            worn[s:s + int(rng.integers(1, 3))] = 0
    for _ in range(rng.poisson(2)):           # forgot to wear
        s = int(rng.integers(0, H))
        worn[s:s + int(rng.integers(3, 13))] = 0
    hosp = crisis.astype(bool) & (rng.random(H) < 0.35)
    worn[hosp] = 0

    # App check-ins (09:00 and 21:00) – self-reported pain, not a model input
    pain = np.full(H, np.nan)
    chk = np.isin(hod, [9, 21]) & (rng.random(H) < 0.7)
    pain_level = np.clip(rng.normal(1.5, 1.0, H) + 2.5 * p + 6.5 * crisis + 3 * recovery, 0, 10)
    pain[chk] = np.round(pain_level[chk])

    # ------------------------------------------------------------------ labels
    onset_flag = np.zeros(H, dtype=int)
    onset_flag[onsets] = 1
    hours_to_next = np.full(H, np.nan)
    nxt = None
    for i in range(H - 1, -1, -1):
        if onset_flag[i]:
            nxt = i
        hours_to_next[i] = (nxt - i) if nxt is not None else np.nan
    voc_active = crisis.astype(int)
    label = ((hours_to_next > 0) & (hours_to_next <= C.HORIZON_H) & (voc_active == 0)).astype(int)
    exclude = ((in_crisis > 0) | (np.arange(H) < C.CALIBRATION_HOURS)).astype(int)

    df = pd.DataFrame({
        "patient_id": pid, "timestamp": ts, "hour_index": np.arange(H),
        "ppg_pi_520": pi520, "ppg_pi_660": pi660, "ppg_pi_940": pi940, "ppg_pi_810": pi810,
        "ror_660_940": ror, "spo2_est": spo2, "transit_delay_810_ms": transit,
        "heart_rate_bpm": hr, "hrv_rmssd_ms": rmssd, "eda_tonic_us": eda, "eda_scr_per_min": scr,
        "skin_temp_c": skin, "ambient_temp_c": amb, "steps_per_hour": steps, "signal_quality": quality,
        "device_worn": worn.astype(int), "sleep_detected": sleep.astype(int),
        "pain_score_app": pain,
        "voc_onset": onset_flag, "voc_active": voc_active,
        "hours_to_next_onset": hours_to_next, "label_voc_next_24h": label,
        "eval_exclude": exclude,
        "sim_prodrome_intensity": np.round(p, 3), "sim_aborted_episode": (ab > 0).astype(int),
        "sim_cold_exposure": cold.astype(int), "sim_infection": infection.astype(int),
    })
    wearable = ["ppg_pi_520", "ppg_pi_660", "ppg_pi_940", "ppg_pi_810", "ror_660_940", "spo2_est",
                "transit_delay_810_ms", "heart_rate_bpm", "hrv_rmssd_ms", "eda_tonic_us",
                "eda_scr_per_min", "skin_temp_c", "steps_per_hour", "signal_quality"]
    df.loc[worn == 0, wearable] = np.nan
    rnd = {"ppg_pi_520": 3, "ppg_pi_660": 3, "ppg_pi_940": 3, "ppg_pi_810": 3, "ror_660_940": 4,
           "spo2_est": 1, "transit_delay_810_ms": 1, "heart_rate_bpm": 1, "hrv_rmssd_ms": 1,
           "eda_tonic_us": 3, "eda_scr_per_min": 2, "skin_temp_c": 2, "ambient_temp_c": 1,
           "steps_per_hour": 0, "signal_quality": 2}
    df = df.round(rnd)

    profile = dict(
        patient_id=pid, age_years=age, sex=sex, genotype=genotype, region=region,
        hydroxyurea=int(hu), hbf_percent=round(hbf, 1), baseline_hb_g_dl=hb, bmi=bmi,
        voc_episodes_prior_12m=max(3, prior_ed + prior_adm), ed_visits_prior_12m=prior_ed,
        admissions_prior_12m=prior_adm, monitoring_start=ts[0].date(), monitoring_days=C.MONITOR_DAYS,
        voc_events_during_monitoring=len(onsets), wear_time_percent=round(100 * worn.mean(), 1),
        sim_signature_strength=round(strength, 3),
    )
    return profile, df, events


def generate_cohort(seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    profiles, frames, events = [], [], []
    for i in range(1, C.N_PATIENTS + 1):
        pr, df, ev = make_patient(i, rng)
        profiles.append(pr)
        frames.append(df)
        events.extend(ev)
    patients = pd.DataFrame(profiles)
    hourly = pd.concat(frames, ignore_index=True)
    events = pd.DataFrame(events)

    # patient-level split, stratified loosely by number of events
    order = rng.permutation(len(patients))
    split = np.empty(len(patients), dtype=object)
    split[order[:C.N_TRAIN]] = "train"
    split[order[C.N_TRAIN:C.N_TRAIN + C.N_VAL]] = "validation"
    split[order[C.N_TRAIN + C.N_VAL:]] = "test"
    patients["data_split"] = split
    return patients, hourly, events


DICTIONARY = [
    # table, column, unit, description, used_as_model_input
    ("patients", "patient_id", "", "Pseudonymised patient code", "no"),
    ("patients", "age_years", "years", "Age at enrolment", "no"),
    ("patients", "sex", "F/M", "Sex", "no"),
    ("patients", "genotype", "", "SCD genotype: HbSS, HbSβ0, HbSβ+, HbSC", "no"),
    ("patients", "region", "", "Region of residence in Saudi Arabia", "no"),
    ("patients", "hydroxyurea", "0/1", "On hydroxyurea therapy", "no"),
    ("patients", "hbf_percent", "%", "Fetal haemoglobin", "no"),
    ("patients", "baseline_hb_g_dl", "g/dL", "Steady-state haemoglobin", "no"),
    ("patients", "bmi", "kg/m²", "Body-mass index", "no"),
    ("patients", "voc_episodes_prior_12m", "count", "VOC episodes in the 12 months before enrolment (inclusion ≥ 3)", "no"),
    ("patients", "ed_visits_prior_12m", "count", "SCD-related ED visits in prior 12 months", "no"),
    ("patients", "admissions_prior_12m", "count", "SCD-related admissions in prior 12 months", "no"),
    ("patients", "monitoring_start", "date", "First day of wristband wear", "no"),
    ("patients", "voc_events_during_monitoring", "count", "VOC onsets observed during the 60-day study", "no"),
    ("patients", "wear_time_percent", "%", "Share of hours the device was worn", "no"),
    ("patients", "data_split", "", "train / validation / test (split by patient)", "no"),
    ("patients", "sim_signature_strength", "0–1", "SIMULATION ONLY: how strongly this patient expresses a prodrome", "no"),
    ("wearable_hourly", "timestamp", "datetime", "Start of the hourly aggregation window", "no"),
    ("wearable_hourly", "ppg_pi_520", "%", "Green (520 nm) PPG perfusion index (AC/DC×100): superficial capillary perfusion", "yes"),
    ("wearable_hourly", "ppg_pi_660", "%", "Red (660 nm) PPG perfusion index", "yes"),
    ("wearable_hourly", "ppg_pi_940", "%", "Near-infrared (940 nm) PPG perfusion index", "yes"),
    ("wearable_hourly", "ppg_pi_810", "%", "Isosbestic (810 nm) PPG perfusion index: oxygenation-independent blood volume", "yes"),
    ("wearable_hourly", "ror_660_940", "ratio", "Ratio-of-ratios (AC660/DC660)/(AC940/DC940): rises with deoxygenation", "yes"),
    ("wearable_hourly", "spo2_est", "%", "SpO2 estimated as 110 − 25 × ratio-of-ratios", "yes"),
    ("wearable_hourly", "transit_delay_810_ms", "ms", "Microvascular pulse transit delay on the 810 nm channel", "yes"),
    ("wearable_hourly", "heart_rate_bpm", "bpm", "Mean heart rate", "yes"),
    ("wearable_hourly", "hrv_rmssd_ms", "ms", "Heart-rate variability (RMSSD)", "yes"),
    ("wearable_hourly", "eda_tonic_us", "µS", "Electrodermal activity: tonic skin-conductance level", "yes"),
    ("wearable_hourly", "eda_scr_per_min", "1/min", "Electrodermal activity: skin-conductance responses per minute", "yes"),
    ("wearable_hourly", "skin_temp_c", "°C", "Peripheral (wrist) thermistor temperature", "yes"),
    ("wearable_hourly", "ambient_temp_c", "°C", "Local outdoor temperature from the paired phone/weather service", "yes"),
    ("wearable_hourly", "steps_per_hour", "steps", "Accelerometer step count", "yes"),
    ("wearable_hourly", "signal_quality", "0–1", "Accelerometer-referenced PPG signal-quality index", "yes"),
    ("wearable_hourly", "device_worn", "0/1", "Wristband on-skin detection (0 → all wearable channels empty)", "yes (as mask)"),
    ("wearable_hourly", "sleep_detected", "0/1", "Sleep detected by the wristband", "no"),
    ("wearable_hourly", "pain_score_app", "0–10", "Self-reported pain from app check-ins (09:00, 21:00)", "no"),
    ("wearable_hourly", "voc_onset", "0/1", "1 at the hour a VOC began", "label"),
    ("wearable_hourly", "voc_active", "0/1", "1 while a VOC is ongoing", "label"),
    ("wearable_hourly", "hours_to_next_onset", "h", "Hours until the next VOC onset", "label"),
    ("wearable_hourly", "label_voc_next_24h", "0/1", "PREDICTION TARGET: a VOC starts within the next 24 h", "target"),
    ("wearable_hourly", "eval_exclude", "0/1", "Hour excluded from evaluation (calibration week, crisis, 72 h recovery)", "no"),
    ("wearable_hourly", "sim_prodrome_intensity", "0–1", "SIMULATION ONLY: true pre-crisis signal intensity", "no"),
    ("wearable_hourly", "sim_aborted_episode", "0/1", "SIMULATION ONLY: prodrome-like episode that did not become a VOC", "no"),
    ("wearable_hourly", "sim_cold_exposure", "0/1", "SIMULATION ONLY: hidden cold-exposure episode", "no"),
    ("wearable_hourly", "sim_infection", "0/1", "SIMULATION ONLY: hidden intercurrent infection", "no"),
    ("voc_events", "event_id", "", "Event code", "no"),
    ("voc_events", "onset_time", "datetime", "VOC onset (first hour of pain requiring treatment)", "label"),
    ("voc_events", "crisis_duration_h", "h", "Duration of the crisis", "no"),
    ("voc_events", "care_setting", "", "home-managed / ED visit / admitted", "no"),
    ("voc_events", "likely_trigger", "", "Trigger recorded in the 24 h before onset", "no"),
    ("voc_events", "peak_pain_score", "0–10", "Peak self-reported pain", "no"),
    ("voc_events", "sim_prodrome_lead_h", "h", "SIMULATION ONLY: true length of the pre-crisis signature", "no"),
    ("voc_events", "sim_prodrome_intensity", "0–1", "SIMULATION ONLY: true prodrome strength", "no"),
]


def save(patients, hourly, events):
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    patients.to_csv(C.DATA_DIR / "patients.csv", index=False)
    events.to_csv(C.DATA_DIR / "voc_events.csv", index=False)
    hourly.to_csv(C.DATA_DIR / "wearable_hourly.csv.gz", index=False, compression="gzip")
    dic = pd.DataFrame(DICTIONARY, columns=["table", "column", "unit", "description", "model_role"])
    dic.to_csv(C.DATA_DIR / "data_dictionary.csv", index=False)

    db = C.DATA_DIR / "sicklepredict.db"
    if db.exists():
        db.unlink()
    with sqlite3.connect(db) as con:
        p = patients.copy()
        p["monitoring_start"] = p["monitoring_start"].astype(str)
        p.to_sql("patients", con, index=False)
        e = events.copy()
        e["onset_time"] = e["onset_time"].astype(str)
        e.to_sql("voc_events", con, index=False)
        h = hourly.copy()
        h["timestamp"] = h["timestamp"].astype(str)
        h.to_sql("wearable_hourly", con, index=False, chunksize=50000)
        dic.to_sql("data_dictionary", con, index=False)
        con.execute("CREATE INDEX idx_hourly_patient ON wearable_hourly(patient_id, hour_index)")
        con.execute("CREATE INDEX idx_events_patient ON voc_events(patient_id)")


def main():
    patients, hourly, events = generate_cohort()
    save(patients, hourly, events)
    print(f"patients: {len(patients)} | hourly rows: {len(hourly):,} | VOC events: {len(events)}")
    print(patients.groupby("data_split")["voc_events_during_monitoring"].agg(["count", "sum"]))
    return patients, hourly, events


if __name__ == "__main__":
    main()
