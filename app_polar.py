import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

st.set_page_config(page_title="StressSys Polar", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
POLAR_DIR = PROJECT_ROOT / "data" / "polar"
MODEL_PATH = PROJECT_ROOT / "models" / "decision_tree_swell.pkl"

WINDOW_SIZE = 300
STEP_SIZE = 60
WINDOW_MINUTES = 5
DEMO_DELAY_SEC = 0.4


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def list_rr_files():
    return sorted(POLAR_DIR.glob("*_RR.txt"))


def load_rr_file(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if "timestamp" in line.lower():
            continue

        parts = line.split(";")

        try:
            rr_ms = float(parts[-1])

            if 300 <= rr_ms <= 2000:
                rows.append({"rr_ms": rr_ms})
        except Exception:
            continue

    return pd.DataFrame(rows)


def compute_rmssd(rr_ms):
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < 2:
        return None
    return np.sqrt(np.mean(np.diff(rr) ** 2))


def compute_sdrr(rr_ms):
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < 2:
        return None
    return np.std(rr)


def compute_pnn50(rr_ms):
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < 2:
        return None
    diff_rr = np.abs(np.diff(rr))
    return np.sum(diff_rr > 50) / len(diff_rr) * 100


def compute_lf_hf(rr_ms):
    rr_sec = np.asarray(rr_ms, dtype=float) / 1000.0

    if len(rr_sec) < 10 or np.any(rr_sec <= 0):
        return 1.0

    time_axis = np.cumsum(rr_sec)
    time_axis = time_axis - time_axis[0]

    if time_axis[-1] <= 1:
        return 1.0

    fs = 4.0
    interp_time = np.arange(0, time_axis[-1], 1 / fs)

    if len(interp_time) < 16:
        return 1.0

    rr_interp = np.interp(interp_time, time_axis, rr_sec)
    rr_interp = rr_interp - np.mean(rr_interp)

    freqs, psd = welch(
        rr_interp,
        fs=fs,
        nperseg=min(256, len(rr_interp))
    )

    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.40)

    if not np.any(lf_band) or not np.any(hf_band):
        return 1.0

    lf_power = np.trapezoid(psd[lf_band], freqs[lf_band])
    hf_power = np.trapezoid(psd[hf_band], freqs[hf_band])

    if hf_power <= 0:
        return 1.0

    return lf_power / hf_power


def extract_features(rr_window):
    rmssd = compute_rmssd(rr_window)
    sdrr = compute_sdrr(rr_window)
    lf_hf = compute_lf_hf(rr_window)
    pnn50 = compute_pnn50(rr_window)
    mean_rr = np.mean(rr_window)

    return {
        "RMSSD": rmssd,
        "SDRR": sdrr,
        "LF_HF": lf_hf,
        "pNN50": pnn50,
        "MEAN_RR": mean_rr
    }


def normalize_by_baseline(current_features, baseline_features):
    normalized = {}

    for key, current_value in current_features.items():
        baseline_value = baseline_features.get(key)

        if baseline_value is None or baseline_value == 0:
            normalized[key] = current_value
        else:
            normalized[key] = current_value / baseline_value

    return normalized


def map_condition_to_state(condition):
    condition = str(condition).lower()

    if "no" in condition or "baseline" in condition or condition == "0":
        return "normal"

    if "time" in condition or "pressure" in condition or condition == "1":
        return "productive_stress"

    if "interrupt" in condition or "stress" in condition or condition == "2":
        return "overload"

    return "normal"


def check_need_break(states):
    if len(states) < 8:
        return False

    return all(state == "productive_stress" for state in states[-8:])


def update_battery(battery, state, need_break):
    if state == "overload":
        battery -= 1.2
    elif state == "productive_stress":
        battery -= 0.4
    else:
        battery += 0.2

    if need_break:
        battery -= 0.8

    return max(0.0, min(100.0, battery))


def show_recommendation(box, battery, state, need_break):
    if battery <= 20:
        box.error("Ресурс майже вичерпано. Рекомендується завершити інтенсивну роботу.")
    elif state == "overload":
        box.warning("OVERLOAD: виявлено перевантаження. Рекомендується перерва.")
    elif need_break:
        box.warning("Продуктивне напруження триває занадто довго. Рекомендується зробити перерву.")
    elif state == "productive_stress":
        box.success("Стан продуктивного стресу. Можна виконувати складні когнітивні задачі.")
    else:
        box.info("Стан стабільний. Можна працювати у звичайному режимі.")


def draw_charts(timeline, battery_chart, rhythm_chart):
    if not timeline:
        return

    time_values = [t["minutes"] for t in timeline]
    battery_values = [t["battery"] for t in timeline]

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(time_values, battery_values)
    ax1.set_ylim(0, 100)
    ax1.set_title("Stress Battery Over Time")
    ax1.set_xlabel("Time, minutes")
    ax1.set_ylabel("Battery")
    ax1.grid()
    battery_chart.pyplot(fig1)
    plt.close(fig1)

    state_map_plot = {
        "normal": 0,
        "productive_stress": 1,
        "overload": 2
    }

    state_values = [state_map_plot[t["state"]] for t in timeline]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.step(time_values, state_values, where="post")
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["normal", "productive", "overload"])
    ax2.set_title("Ultradian Stress Rhythm — Polar H10")
    ax2.set_xlabel("Time, minutes")
    ax2.set_ylabel("State")
    ax2.grid()

    for t in timeline:
        if t["need_break"]:
            ax2.axvline(t["minutes"], linestyle="--", alpha=0.4)

    rhythm_chart.pyplot(fig2)
    plt.close(fig2)


model_data = load_model()
model = model_data["model"]
scaler = model_data.get("scaler")
features = model_data["features"]
accuracy = model_data.get("test_accuracy", model_data.get("accuracy", 0))

st.title("StressSys — Polar H10 HRV monitoring")

rr_files = list_rr_files()

if not rr_files:
    st.error("У папці data/polar/ немає RR-файлів.")
    st.stop()

file_options = ["Об'єднати всі RR-файли"] + [f.name for f in rr_files]

selected_file = st.selectbox(
    "Джерело Polar RR-даних",
    file_options
)

if selected_file == "Об'єднати всі RR-файли":
    rr_frames = []

    for file in rr_files:
        part = load_rr_file(file)
        part["source"] = file.name
        rr_frames.append(part)

    rr_df = pd.concat(rr_frames, ignore_index=True)
    st.info(f"Завантажено всі RR-файли: {len(rr_files)}")
else:
    file_path = POLAR_DIR / selected_file
    rr_df = load_rr_file(file_path)
    st.info(f"Завантажено файл: {selected_file}")

if rr_df.empty:
    st.error("RR-дані не прочитались.")
    st.stop()

st.caption(
    "Система працює у режимі offline playback: Polar RR-файл програється як потік wearable-даних."
)

st.divider()

top1, top2, top3, top4 = st.columns(4)

top1.metric("Модель SWELL accuracy", f"{accuracy:.2%}")

battery_box = top2.empty()
state_box = top3.empty()
time_box = top4.empty()

battery_box.metric("Поточна батарея", "100.0%")
state_box.metric("Поточний стан", "очікування")
time_box.metric("Симульований час", "0 хв")

st.divider()

st.subheader("Фаза 1 — персональна калібровка")

if len(rr_df) < WINDOW_SIZE:
    st.error("Недостатньо RR-даних для одного HRV-вікна.")
    st.stop()

calibration_rr = rr_df["rr_ms"].values[:WINDOW_SIZE]
calibration_features = extract_features(calibration_rr)

if calibration_features is None:
    st.error("Не вдалося сформувати baseline HRV.")
    st.stop()

st.success(
    f"Калібровка завершена: перше HRV-вікно використано як baseline "
    f"({WINDOW_MINUTES} хв умовного фізіологічного часу)."
)

with st.expander("Показати baseline HRV"):
    st.write(pd.DataFrame([calibration_features]))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Стрес-батарея за сесію")
    battery_chart = st.empty()

with right:
    st.subheader("Ультрадіанні ритми")
    rhythm_chart = st.empty()

st.divider()

recommendation_box = st.empty()
progress_bar = st.progress(0)

start_button = st.button("▶ Запустити Polar playback")

if not start_button:
    recommendation_box.info("Натисніть кнопку запуску для аналізу Polar RR-даних.")
    st.stop()

rr_all = rr_df["rr_ms"].values

states = []
timeline = []
battery = 100.0

max_steps = max(0, (len(rr_all) - WINDOW_SIZE) // STEP_SIZE)

if max_steps == 0:
    st.error("Недостатньо даних для rolling-аналізу.")
    st.stop()

for step in range(max_steps):
    start = step * STEP_SIZE
    end = start + WINDOW_SIZE

    rr_window = rr_all[start:end]

    feature_dict = extract_features(rr_window)

    if feature_dict is None:
        continue

    normalized_features = normalize_by_baseline(
        feature_dict,
        calibration_features
    )

    X_live = pd.DataFrame([normalized_features])
    X_live = X_live[features]

    if scaler is not None:
        X_live_scaled = scaler.transform(X_live)
        pred_condition = model.predict(X_live_scaled)[0]
    else:
        pred_condition = model.predict(X_live)[0]

    state = map_condition_to_state(pred_condition)

    states.append(state)

    need_break = check_need_break(states)
    battery = update_battery(battery, state, need_break)

    simulated_minutes = (step + 1) * WINDOW_MINUTES

    timeline.append({
        "step": step,
        "minutes": simulated_minutes,
        "state": state,
        "battery": round(battery, 2),
        "need_break": need_break,
        "prediction": str(pred_condition),
        "rmssd": round(feature_dict["RMSSD"], 2),
        "sdrr": round(feature_dict["SDRR"], 2),
        "lf_hf": round(feature_dict["LF_HF"], 2),
        "pnn50": round(feature_dict["pNN50"], 2),
        "mean_rr": round(feature_dict["MEAN_RR"], 2),
        "rmssd_norm": round(normalized_features["RMSSD"], 3),
        "sdrr_norm": round(normalized_features["SDRR"], 3),
        "lf_hf_norm": round(normalized_features["LF_HF"], 3),
        "pnn50_norm": round(normalized_features["pNN50"], 3),
        "mean_rr_norm": round(normalized_features["MEAN_RR"], 3),
    })

    battery_box.metric("Поточна батарея", f"{battery:.1f}%")
    state_box.metric("Поточний стан", state)
    time_box.metric("Симульований час", f"{simulated_minutes} хв")

    draw_charts(timeline, battery_chart, rhythm_chart)
    show_recommendation(recommendation_box, battery, state, need_break)

    progress_bar.progress((step + 1) / max_steps)

    time.sleep(DEMO_DELAY_SEC)

results_df = pd.DataFrame(timeline)

output_path = PROJECT_ROOT / "data" / "polar_predictions_ui.csv"
results_df.to_csv(output_path, index=False)

st.divider()
st.subheader("Результати Polar playback")
st.dataframe(results_df)

st.success(f"Результати збережено: {output_path}")