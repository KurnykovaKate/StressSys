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

POLAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "polar"
    / "Polar_H10_78887921_20210712_232912_RR.txt"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "decision_tree_swell.pkl"
)


WINDOW_SIZE = 300
STEP_SIZE = 60


def extract_features(rr):

    rmssd = compute_rmssd(rr)
    sdnn = compute_sdnn(rr)
    pnn50 = compute_pnn50(rr)
    mean_hr = compute_mean_hr(rr)

    # временная заглушка
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

    rr_all = df["rr_ms"].values

    print("RR count:", len(rr_all))

    print("\nROLLING ANALYSIS\n")

    results = []

    start = 0

    while start + WINDOW_SIZE < len(rr_all):

        rr_window = rr_all[start:start + WINDOW_SIZE]

        features = extract_features(rr_window)

        X = pd.DataFrame([features])

        prediction = model.predict(X)[0]

        result = {
            "window_start": start,
            "window_end": start + WINDOW_SIZE,
            "prediction": prediction,
            "rmssd": round(features["RMSSD"], 2),
            "sdrr": round(features["SDRR"], 2),
            "pnn50": round(features["pNN50"], 2),
            "mean_rr": round(features["MEAN_RR"], 2)
        }

        results.append(result)

        print(
            f"[{start}:{start + WINDOW_SIZE}] "
            f"{prediction} | "
            f"RMSSD={result['rmssd']} | "
            f"SDRR={result['sdrr']} | "
            f"pNN50={result['pnn50']}"
        )

        start += STEP_SIZE

    results_df = pd.DataFrame(results)

    output_path = PROJECT_ROOT / "data" / "polar_predictions.csv"

    results_df.to_csv(output_path, index=False)

    print("\nSaved predictions:")
    print(output_path)