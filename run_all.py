"""
SicklePredict prototype — run everything with one command:

    python run_all.py

1. generates the synthetic patient database      -> data/
2. trains and evaluates the 5 models             -> outputs/models/
3. writes the files the results dashboard reads  -> outputs/dashboard_data/
"""
from sicklepredict import excel_export, simulate, train

if __name__ == "__main__":
    print("=== Step A: synthetic database ===")
    simulate.main()
    excel_export.main()
    print("\n=== Step B: model training & evaluation ===")
    train.main()
