# SicklePredict: Prototype (MED-LKD7SS)

**A multispectral wearable and AI model for early warning of vaso-occlusive crises (VOC) in sickle cell disease.**
MEDATHON 2026 · Filter 2 · Track 3: Medical Tools & Devices Innovation

> ⚠️ **All patient data in this repository is simulated.** The pre-crisis physiological signature is an assumption written into the simulator. The results show that the full pipeline works, from signals to alerts. They are **not** evidence of clinical accuracy, which requires the IRB-approved pilot study described in the proposal.

---

## What is in this repository

| Folder | Contents |
|---|---|
| `sicklepredict/` | Python source: data simulator, personal-baseline features, Bi-LSTM + temporal attention (written from scratch in NumPy), baselines, evaluation, prediction |
| `data/` | Synthetic database: 200 patients × 60 days of hourly wristband data (288,000 rows, 234 VOC events), Excel copy and data dictionary |
| `outputs/` | Trained model weights, run settings, and every results table behind the slides and report |
| `watch_simulator/` | Interactive smartwatch-app prototype (HTML/CSS/JS, Arabic/English) that replays test patients with the trained model's real risk scores |
| `docs/` | Results report (Word) and an Arabic user guide |

## Results on 40 held-out test patients (simulated data)

| Model | AUROC (95% CI) | Crises warned in advance | Median lead time | False alerts / patient-week |
|---|---|---|---|---|
| Rule-based threshold | 0.58 (0.50–0.66) | 32/39 (82%) | 16.5 h | 9.19 |
| Random forest (prior-work style) | 0.75 (0.68–0.81) | 27/39 (69%) | 14 h | 1.30 |
| Bi-LSTM, population baseline | 0.65 (0.58–0.73) | 8/39 (21%) | 9 h | 0.47 |
| Bi-LSTM, consumer sensors only | 0.84 (0.78–0.89) | 27/39 (69%) | 10 h | 0.15 |
| **SicklePredict (proposed)** | **0.85 (0.80–0.90)** | **33/39 (85%)** | **12 h** | **0.13** |

Patient-level split (120 train / 40 validation / 40 test). The alert threshold was fixed on validation patients before testing. Confidence intervals use a patient-level bootstrap.

## How to run

Requires Python 3.10+ and no GPU.

```bash
pip install -r requirements.txt
python run_all.py          # regenerates the data, trains all 5 models, writes outputs/ (~6 min)
```

Score new wristband data (same columns as `data/wearable_hourly.csv.gz`, at least 8 days per patient):

```bash
python -m sicklepredict.predict new_data.csv risk_scores.csv
```

**Watch simulator:** open `watch_simulator/index.html` in Chrome, Safari or Edge. To get a live link, enable **GitHub Pages** (Settings → Pages → Branch: `main`) and open `.../watch_simulator/`.

All assumptions (cohort size, prodrome window, model settings, alert rules) are in `sicklepredict/config.py`.

## Method in brief

1. **Sensors (hourly):** PPG perfusion index at 520 / 660 / 940 / 810 nm (isosbestic), ratio-of-ratios and SpO₂, 810 nm microvascular transit delay, heart rate, HRV, tonic and phasic EDA, wrist temperature, steps, signal quality, outdoor temperature.
2. **Personal baseline:** each patient's first 7 days give their own median and spread for four times of day. Later hours are expressed as personal deviations.
3. **Model:** a bidirectional LSTM reads the last 24 h, and temporal attention weights each hour. The output is calibrated to the probability of a VOC starting within the next 24 h.
4. **Alerting:** an alert fires when risk stays above the threshold for 2 consecutive hours, at most once every 12 h.

## Team

شهد القواسمي · جهاد القواسمي · روان الحربي · بشرى النجلات

---

## بالعربي باختصار

نموذج أولي برمجي لمشروع **SicklePredict**، سوار ذكي يتنبأ مبكرًا بنوبات الانسداد الوعائي لدى مرضى فقر الدم المنجلي. يحتوي المستودع على:
- قاعدة بيانات اصطناعية لـ 200 مريض.
- كود تدريب نموذج Bi-LSTM مع الانتباه (Attention).
- نتائج الاختبار على 40 مريضًا لم يرهم النموذج أثناء التدريب.
- محاكي تفاعلي لتطبيق الساعة.

**جميع البيانات محاكاة**، والنتائج تثبت جاهزية النظام وليس دقته السريرية. دليل التشغيل بالعربي موجود في `docs/`.
