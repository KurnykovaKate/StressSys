import pickle
import warnings
from pathlib import Path

from hrv_features import get_rr_intervals, compute_rmssd, compute_sdnn
from windowing import create_windows

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "WESAD" / "WESAD"


def load_subject(subject_id: str):
    subject_id = subject_id.upper()
    file_path = DATASET_PATH / subject_id / f"{subject_id}.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    return data


if __name__ == "__main__":
    subject = load_subject("S2")

    ecg = subject["signal"]["chest"]["ECG"].flatten()
    labels = subject["label"]

    print("Дані завантажені успішно")
    print("Ключі:", subject.keys())
    print("ECG довжина:", len(ecg))
    print("Labels довжина:", len(labels))
    print("ECG форма:", subject["signal"]["chest"]["ECG"].shape)

    ecg_windows, label_windows = create_windows(ecg, labels)

    print("Кількість вікон:", len(ecg_windows))
    print("Мітка першого вікна:", label_windows[0])
    print("Довжина першого вікна:", len(ecg_windows[0]))

    first_window = ecg_windows[0]

    rr = get_rr_intervals(first_window)
    rmssd = compute_rmssd(rr)
    sdnn = compute_sdnn(rr)

    print("RR інтервали:", rr[:5])
    print("RMSSD:", rmssd)
    print("SDNN:", sdnn)