"""Train/evaluate the user-collected drowsiness ML classifier."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from features import FEATURE_NAMES

MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"


def make_model():
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=16,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def best_threshold(y_true, drowsy_probability):
    """Learn the probability cutoff from grouped out-of-fold predictions instead of hard-coding it."""
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(0.30, 0.81, 0.01):
        pred = (drowsy_probability >= t).astype(int)
        score = f1_score(y_true, pred, average="macro")
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit("data.csv not found. Run collect_normal.bat and collect_drowsy.bat first.")
    df = pd.read_csv(DATA_PATH)
    required = FEATURE_NAMES + ["label", "session"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"data.csv is missing columns: {missing}. Delete old data.csv and recollect once.")

    df = df[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if df["label"].nunique() < 2:
        raise SystemExit("Need both normal and drowsy samples.")
    sessions = df["session"].astype(str)
    if sessions.nunique() < 4:
        raise SystemExit("Need at least 4 sessions. Recommended: 3 normal + 3 drowsy.")

    X, y = df[FEATURE_NAMES], df["label"]
    n_splits = min(6, sessions.nunique())
    cv = GroupKFold(n_splits=n_splits)
    model = make_model()

    # Honest grouped cross-validation: a whole recording session stays in one fold.
    pred = cross_val_predict(model, X, y, cv=cv, groups=sessions, n_jobs=1, method="predict")
    prob = cross_val_predict(model, X, y, cv=cv, groups=sessions, n_jobs=1, method="predict_proba")
    # cross_val_predict returns probability columns in estimator.classes_ order.
    # The unfitted base estimator does not have classes_, so derive the order from y.
    classes = sorted(y.unique().tolist())
    d_idx = classes.index(1)
    dprob = prob[:, d_idx]

    threshold, threshold_f1 = best_threshold(y.to_numpy(), dprob)
    threshold_pred = (dprob >= threshold).astype(int)

    print("\n=== GROUPED CROSS-VALIDATION TEST ===")
    print(f"Sessions: {sessions.nunique()} | Samples: {len(df)} | Folds: {n_splits}")
    print(f"Learned AI probability threshold: {threshold:.2f}")
    print(f"Threshold macro F1: {threshold_f1:.3f}")
    print(f"Accuracy: {accuracy_score(y, threshold_pred):.3f}")
    print(f"Macro F1:  {f1_score(y, threshold_pred, average='macro'):.3f}")
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y, threshold_pred, labels=[0, 1]))
    print(classification_report(y, threshold_pred, target_names=["normal", "drowsy"], zero_division=0))

    final_model = make_model()
    final_model.fit(X, y)
    bundle = {
        "model": final_model,
        "threshold": float(threshold),
        "features": FEATURE_NAMES,
        "version": 2,
    }
    joblib.dump(bundle, MODEL_PATH)
    print("Saved model.pkl — AI mode enabled.")
    print("Top learned feature importance:")
    for name, imp in sorted(zip(FEATURE_NAMES, final_model.feature_importances_), key=lambda z: z[1], reverse=True):
        print(f"  {name:18s} {imp:.3f}")


if __name__ == "__main__":
    main()
