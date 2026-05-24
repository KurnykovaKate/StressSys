from pathlib import Path

import numpy as np
import pandas as pd

from hrv_features import get_rr_intervals, compute_rmssd, compute_sdnn
from step_data import load_subject
from windowing import create_windows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "features.csv"

SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6", "S7", "S8",
    "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"
]


def build_features_for_subject(subject_id):
    subject = load_subject(subject_id)

    ecg = subject["signal"]["chest"]["ECG"].flatten()
    labels = subject["label"]

    ecg_windows, label_windows = create_windows(ecg, labels)

    rows = []

    for window_id, (window, label) in enumerate(zip(ecg_windows, label_windows)):
        rr = get_rr_intervals(window)

        rmssd = compute_rmssd(rr)
        sdnn = compute_sdnn(rr)

        if rmssd is None or sdnn is None or len(rr) == 0:
            continue

        mean_hr = 60 / np.mean(rr)

        rows.append({
            "subject": subject_id,
            "window_id": window_id,
            "label": int(label),
            "rmssd": rmssd * 1000,
            "sdnn": sdnn * 1000,
            "mean_hr": mean_hr
        })

    return rows


if __name__ == "__main__":
    all_rows = []

    for subject_id in SUBJECTS:
        print(f"Обробляю {subject_id}...")
        rows = build_features_for_subject(subject_id)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Готово")
    print("Збережено в:", OUTPUT_PATH)
    print("Розмір таблиці:", df.shape)
    print(df.head())