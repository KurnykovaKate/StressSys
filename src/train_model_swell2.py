from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "swell" / "archive" / "hrv dataset" / "data" / "final" / "train.csv"
TEST_PATH  = PROJECT_ROOT / "data" / "swell" / "archive" / "hrv dataset" / "data" / "final" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "decision_tree_swell.pkl"

# ── 1. Завантаження ────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Колонки:", list(train_df.columns))

label_col = "condition"

# ── 2. Ознаки — тільки ті що вимірює носимий пристрій ─────────
features = ["RMSSD", "SDRR", "LF_HF", "pNN50", "MEAN_RR"]

# перевірка яких ознак немає в датасеті
available = [f for f in features if f in train_df.columns]
missing   = [f for f in features if f not in train_df.columns]

if missing:
    print(f"\nВідсутні ознаки (пропускаємо): {missing}")

features = available
print(f"\nВикористані ознаки: {features}")

X_train = train_df[features]
y_train = train_df[label_col]

X_test = test_df[features]
y_test = test_df[label_col]

# ── 3. Масштабування ───────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Підбір гіперпараметрів ──────────────────────────────────
param_grid = {
    "max_depth":        [4, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 3, 5],
    "criterion":        ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# ── 5. Результати ──────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\nНайкращі гіперпараметри:")
print(grid_search.best_params_)

print(f"\nCV accuracy:   {grid_search.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred))

print("\nПравила дерева рішень:")
print(export_text(model, feature_names=features))

# ── 6. Збереження ──────────────────────────────────────────────
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({
    "model":       model,
    "scaler":      scaler,
    "features":    features,
    "cv_accuracy": grid_search.best_score_,
    "test_accuracy": accuracy_score(y_test, y_pred),
    "best_params": grid_search.best_params_
}, MODEL_PATH)
print("\nМодель збережена:", MODEL_PATH)

# ── 7. Графіки ─────────────────────────────────────────────────
# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix — Decision Tree SWELL")
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.plot(kind="bar")
plt.title("Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Tree Visualization
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=features,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Decision Tree — SWELL")
plt.tight_layout()
plt.show()