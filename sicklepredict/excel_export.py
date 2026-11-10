"""Readable Excel copy of the synthetic database (dictionary, patients, events, sample week)."""
from __future__ import annotations

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import config as C


def main():
    d = C.DATA_DIR
    dic = pd.read_csv(d / "data_dictionary.csv")
    patients = pd.read_csv(d / "patients.csv")
    events = pd.read_csv(d / "voc_events.csv")
    hourly = pd.read_csv(d / "wearable_hourly.csv.gz")
    # one week around the first event of a test patient, as a readable example
    ex = events.merge(patients[["patient_id", "data_split"]]).query("data_split == 'test'").iloc[0]
    s = max(0, int(ex.onset_hour_index) - 120)
    sample = hourly[(hourly.patient_id == ex.patient_id) & hourly.hour_index.between(s, s + 167)]

    out = d / "SicklePredict_Synthetic_Database.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame({"SicklePredict synthetic database": [
            "SIMULATED DATA – not real patients. Built to develop and test the prototype pipeline.",
            f"{len(patients)} patients × {C.MONITOR_DAYS} days of hourly wristband data = {len(hourly):,} rows; {len(events)} VOC events.",
            "Sheets: Data_Dictionary, Patients, VOC_Events, Sample_Week (one patient, 7 days around a VOC).",
            "Full hourly data: wearable_hourly.csv.gz and sicklepredict.db (SQLite) in the same folder.",
            "Columns starting with sim_ are simulation ground truth and would not exist in real data.",
        ]}).to_excel(xw, sheet_name="README", index=False)
        dic.to_excel(xw, sheet_name="Data_Dictionary", index=False)
        patients.to_excel(xw, sheet_name="Patients", index=False)
        events.to_excel(xw, sheet_name="VOC_Events", index=False)
        sample.to_excel(xw, sheet_name="Sample_Week", index=False)

    wb = load_workbook(out)
    head = PatternFill("solid", fgColor="7A1F2B")
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head
            c.alignment = Alignment(vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"
        for i, col in enumerate(ws.columns, 1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200])
            ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 90 if ws.title == "README" else 48)
    wb.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
