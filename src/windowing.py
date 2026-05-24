SAMPLING_RATE = 700  # Hz
WINDOW_SEC = 60


def create_windows(ecg, labels):
    window_size = SAMPLING_RATE * WINDOW_SEC

    ecg_windows = []
    label_windows = []

    for i in range(0, len(ecg) - window_size, window_size):
        window = ecg[i:i + window_size]
        label = labels[i]

        ecg_windows.append(window)
        label_windows.append(label)

    return ecg_windows, label_windows