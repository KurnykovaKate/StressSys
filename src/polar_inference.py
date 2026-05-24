from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from polar_features import (
    load_rr_file,
    compute_rmssd,
    compute_sdnn,
    compute_pnn50,
    compute_mean_hr
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RR файл Polar
POLAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "polar"
    / "Polar_H10_78887921_20210712_232912_RR.txt"
)

# Твоя обученная модель SWELL
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "decision_tree_swell.pkl"
)


def extract_features(rr):

    rmssd = compute_rmssd(rr)
    sdnn = compute_sdnn(rr)
    pnn50 = compute_pnn50(rr)
    mean_hr = compute_mean_hr(rr)

    # LF/HF пока заглушка
    # потом добавим отдельно
    lf_hf = 1.0

    return {
        "RMSSD": rmssd,
        "SDRR": sdnn,
        "LF_HF": lf_hf,
        "pNN50": pnn50,
        "MEAN_RR": np.mean(rr)
    }


if __name__ == "__main__":

    print("Loading model...")

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]

    print("Model loaded.")

    print("\nLoading Polar RR...")

    df = load_rr_file(POLAR_PATH)

    rr = df["rr_ms"].values

    print("RR count:", len(rr))

    # берем первые 5 минут примерно
    # 300-500 RR уже достаточно для теста
    rr_window = rr[:500]

    features = extract_features(rr_window)

    print("\nEXTRACTED FEATURES:")
    for k, v in features.items():
        print(k, ":", round(v, 2))

    X = pd.DataFrame([features])

    prediction = model.predict(X)[0]

    print("\nMODEL PREDICTION:")
    print(prediction)