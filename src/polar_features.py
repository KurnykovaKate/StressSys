from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POLAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "polar"
    / "Polar_H10_78887921_20210712_232912_RR.txt"
)


def load_rr_file(path):
    """
    Читает RR.txt от Polar H10

    Формат Polar может отличаться:
    timestamp;sensor timestamp;rr
    или
    timestamp;...;rr_ms

    Поэтому берем последнее число в строке.
    """

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Total lines:", len(lines))

    # покажем первые строки файла
    print("\nFIRST LINES:")
    for line in lines[:5]:
        print(line.strip())

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # пропускаем заголовки
        if "timestamp" in line.lower():
            continue

        parts = line.split(";")

        try:
            # берем последнее значение как RR
            rr_ms = float(parts[-1])

            # фильтрация мусора
            if rr_ms < 300 or rr_ms > 2000:
                continue

            rows.append({
                "rr_ms": rr_ms
            })

        except:
            continue

    df = pd.DataFrame(rows)

    print("\nParsed RR rows:", len(df))

    return df


def compute_rmssd(rr):
    rr = np.asarray(rr, dtype=float)

    if len(rr) < 2:
        return None

    diff_rr = np.diff(rr)

    return np.sqrt(np.mean(diff_rr ** 2))


def compute_sdnn(rr):
    rr = np.asarray(rr, dtype=float)

    if len(rr) < 2:
        return None

    return np.std(rr)


def compute_pnn50(rr):
    rr = np.asarray(rr, dtype=float)

    if len(rr) < 2:
        return None

    diff_rr = np.abs(np.diff(rr))

    return np.sum(diff_rr > 50) / len(diff_rr) * 100


def compute_mean_hr(rr):
    rr = np.asarray(rr, dtype=float)

    mean_rr = np.mean(rr)

    if mean_rr <= 0:
        return None

    return 60000 / mean_rr


if __name__ == "__main__":

    print("Loading Polar RR file...\n")

    df = load_rr_file(POLAR_PATH)

    if df.empty:
        raise ValueError(
            "\nНе удалось прочитать RR данные.\n"
            "Проверь формат RR.txt."
        )

    print("\nData preview:")
    print(df.head())

    rr = df["rr_ms"].values

    print("\nRR count:", len(rr))

    rmssd = compute_rmssd(rr)
    sdnn = compute_sdnn(rr)
    pnn50 = compute_pnn50(rr)
    mean_hr = compute_mean_hr(rr)

    print("\nHRV FEATURES")
    print("RMSSD :", round(rmssd, 2))
    print("SDNN  :", round(sdnn, 2))
    print("pNN50 :", round(pnn50, 2))
    print("HR    :", round(mean_hr, 2))