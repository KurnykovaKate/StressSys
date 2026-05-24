import numpy as np
from scipy.signal import find_peaks


def get_rr_intervals(ecg_signal, sampling_rate=700):
    """
    Знаходить R-піки і рахує RR інтервали
    """
    peaks, _ = find_peaks(ecg_signal, distance=sampling_rate*0.6)

    rr_intervals = np.diff(peaks) / sampling_rate  # в секундах
    return rr_intervals


def compute_rmssd(rr_intervals):
    if len(rr_intervals) < 2:
        return None

    diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff ** 2))
    return rmssd


def compute_sdnn(rr_intervals):
    if len(rr_intervals) < 2:
        return None

    return np.std(rr_intervals)