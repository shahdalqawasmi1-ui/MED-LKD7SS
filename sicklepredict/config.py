"""
SicklePredict prototype — central configuration.

Every number that shapes the simulation, the model or the evaluation lives here,
so the whole study can be re-run with different assumptions by editing one file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
DASH_DIR = OUT_DIR / "dashboard_data"

SEED = 2026

# ----------------------------------------------------------------------------
# Synthetic cohort
# ----------------------------------------------------------------------------
N_PATIENTS = 200            # enrolled patients
MONITOR_DAYS = 60           # continuous wear per patient
HOURS = MONITOR_DAYS * 24   # hourly aggregated wearable records
CALIBRATION_HOURS = 7 * 24  # first week = personal-baseline (calibration) period

# Inclusion criterion of the simulated pilot: >= 3 VOC episodes in the prior 12 months
# (an enriched, high-utiliser cohort, as an IRB feasibility study would recruit).
MEAN_VOC_PER_YEAR = 9.5

# Pre-crisis (prodromal) physiological window, hours before VOC onset.
# This is the HYPOTHESIS the proposal wants to test; here it is an assumption.
PRODROME_MIN_H = 12
PRODROME_MAX_H = 48

CRISIS_MIN_DAYS = 2
CRISIS_MAX_DAYS = 7
RECOVERY_H = 72             # post-crisis hours excluded from evaluation
REFRACTORY_DAYS = 7         # no new onset within 7 days after a crisis ends

# ----------------------------------------------------------------------------
# Prediction task
# ----------------------------------------------------------------------------
WINDOW_H = 24               # look-back window fed to the model
HORIZON_H = 24              # predict "VOC onset within the next 24 h"

# Model input channels (hourly aggregates from the wristband + phone context)
SIGNAL_CHANNELS = [
    "ppg_pi_520",            # green perfusion index (%), superficial capillaries
    "ppg_pi_660",            # red perfusion index (%)
    "ppg_pi_940",            # NIR perfusion index (%)
    "ppg_pi_810",            # isosbestic perfusion index (%), total blood volume
    "ror_660_940",           # ratio-of-ratios 660/940 (deoxygenation proxy)
    "spo2_est",              # SpO2 estimated from ratio-of-ratios (%)
    "transit_delay_810_ms",  # microvascular pulse transit delay, 810 nm (ms)
    "heart_rate_bpm",
    "hrv_rmssd_ms",
    "eda_tonic_us",          # skin conductance level (µS)
    "eda_scr_per_min",       # skin conductance responses / min
    "skin_temp_c",           # peripheral thermistor (°C)
    "ambient_temp_c",        # phone / weather context (°C)
    "steps_per_hour",        # accelerometer
    "signal_quality",        # 0–1, accelerometer-referenced artefact index
]
# Extra engineered inputs appended by the feature pipeline
EXTRA_INPUTS = ["worn", "hod_sin", "hod_cos"]

# Channel subsets used for the ablation that mirrors the proposal's hypotheses
CONSUMER_PPG_CHANNELS = [   # what a consumer watch gives: green/red/IR PPG + HR + steps
    "ppg_pi_520", "ppg_pi_660", "ppg_pi_940", "ror_660_940", "spo2_est",
    "heart_rate_bpm", "hrv_rmssd_ms", "steps_per_hour", "signal_quality",
]

MODALITY_GROUPS = {
    "Multi-λ PPG (520/660/940)": ["ppg_pi_520", "ppg_pi_660", "ppg_pi_940", "ror_660_940", "spo2_est"],
    "Isosbestic 810 nm + transit": ["ppg_pi_810", "transit_delay_810_ms"],
    "Electrodermal (EDA)": ["eda_tonic_us", "eda_scr_per_min"],
    "Peripheral thermistor": ["skin_temp_c"],
    "Cardiac (HR / HRV)": ["heart_rate_bpm", "hrv_rmssd_ms"],
    "Activity & context": ["steps_per_hour", "ambient_temp_c", "signal_quality"],
}

# ----------------------------------------------------------------------------
# Data split (by patient — no patient appears in two sets)
# ----------------------------------------------------------------------------
N_TRAIN, N_VAL, N_TEST = 120, 40, 40
NEG_PER_POS_TRAIN = 8       # negative-window subsampling for training

# ----------------------------------------------------------------------------
# Bi-LSTM + temporal attention
# ----------------------------------------------------------------------------
HIDDEN = 24                 # LSTM units per direction
ATTN_DIM = 16
DENSE = 16
EPOCHS = 30
BATCH = 128
LR = 3e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.4
INPUT_NOISE = 0.5          # Gaussian input noise during training (regularisation)
GRAD_CLIP = 5.0
PATIENCE = 6                # early stopping on validation AUPRC

# ----------------------------------------------------------------------------
# Alerting
# ----------------------------------------------------------------------------
ALERT_CONSECUTIVE_H = 2     # risk must stay above threshold 2 h to fire an alert
ALERT_REFRACTORY_H = 12     # one alert per 12 h (avoid alarm fatigue)
N_BOOTSTRAP = 1000          # patient-level bootstrap for 95% CIs
