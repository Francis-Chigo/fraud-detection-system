# ============================================================
# PHASE 4 - Model Evaluation & Tuning
# NIBSS Nigerian Fraud Detection Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

print("=" * 60)
print("  GUARDIAN - NIGERIAN PAYMENT FRAUD DETECTION")
print("  Phase 4: Model Evaluation & Tuning")
print("=" * 60)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("\n⏳ Loading processed data...")
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test  = pd.read_csv("data/processed/X_test.csv")
y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()
print("✅ Data loaded!")

# ── 2. BASELINE XGBOOST ──────────────────────────────────────
print("\n── Baseline XGBoost (before tuning) ──────────────────")
base_model = XGBClassifier(n_estimators=100, random_state=42,
                            eval_metric="logloss", verbosity=0)
base_model.fit(X_train, y_train)
base_pred   = base_model.predict(X_test)
base_proba  = base_model.predict_proba(X_test)[:, 1]
base_auc    = roc_auc_score(y_test, base_proba)
base_report = classification_report(y_test, base_pred, output_dict=True)

print(f"   Precision : {base_report['1']['precision']:.4f}")
print(f"   Recall    : {base_report['1']['recall']:.4f}")
print(f"   F1 Score  : {base_report['1']['f1-score']:.4f}")
print(f"   AUC-ROC   : {base_auc:.4f}")

# ── 3. HYPERPARAMETER TUNING ─────────────────────────────────
print("\n⏳ Tuning XGBoost (this may take 3-5 mins)...")
param_grid = {
    "n_estimators" : [100, 200],
    "max_depth"    : [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample"    : [0.8, 1.0],
}
grid_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
    param_grid, scoring="f1", cv=3, n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n✅ Best parameters found:")
for k, v in grid_search.best_params_.items():
    print(f"   {k} : {v}")

# ── 4. TUNED MODEL ───────────────────────────────────────────
print("\n── Tuned XGBoost (after tuning) ──────────────────────")
tuned_model  = grid_search.best_estimator_
tuned_pred   = tuned_model.predict(X_test)
tuned_proba  = tuned_model.predict_proba(X_test)[:, 1]
tuned_auc    = roc_auc_score(y_test, tuned_proba)
tuned_report = classification_report(y_test, tuned_pred, output_dict=True)

print(f"   Precision : {tuned_report['1']['precision']:.4f}")
print(f"   Recall    : {tuned_report['1']['recall']:.4f}")
print(f"   F1 Score  : {tuned_report['1']['f1-score']:.4f}")
print(f"   AUC-ROC   : {tuned_auc:.4f}")
print("\n── Full Classification Report ────────────────────────")
print(classification_report(y_test, tuned_pred, target_names=["Normal", "Fraud"]))

# ── 5. FEATURE IMPORTANCE ─────────────────────────────────────
feature_names = pd.read_csv("data/processed/X_train.csv").columns.tolist()
importances   = tuned_model.feature_importances_
top_idx       = np.argsort(importances)[::-1][:15]
top_features  = [feature_names[i] for i in top_idx]
top_scores    = importances[top_idx]

# ── 6. VISUALIZATIONS ─────────────────────────────────────────
print("\n⏳ Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Guardian — XGBoost Evaluation & Tuning (NIBSS Nigerian Dataset)",
             fontsize=14, fontweight="bold")

metrics      = ["Precision", "Recall", "F1", "AUC"]
base_scores  = [base_report["1"]["precision"], base_report["1"]["recall"],
                base_report["1"]["f1-score"], base_auc]
tuned_scores = [tuned_report["1"]["precision"], tuned_report["1"]["recall"],
                tuned_report["1"]["f1-score"], tuned_auc]
x = np.arange(len(metrics))
axes[0, 0].bar(x - 0.2, base_scores,  0.35, label="Before Tuning", color="steelblue", alpha=0.85)
axes[0, 0].bar(x + 0.2, tuned_scores, 0.35, label="After Tuning",  color="crimson",   alpha=0.85)
axes[0, 0].set_title("XGBoost: Before vs After Tuning")
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(metrics)
axes[0, 0].set_ylim(0, 1.1); axes[0, 0].legend(); axes[0, 0].set_ylabel("Score")
for i, (b, t) in enumerate(zip(base_scores, tuned_scores)):
    axes[0, 0].text(i - 0.2, b + 0.02, f"{b:.3f}", ha="center", fontsize=8)
    axes[0, 0].text(i + 0.2, t + 0.02, f"{t:.3f}", ha="center", fontsize=8)

cm = confusion_matrix(y_test, tuned_pred)
axes[0, 1].imshow(cm, interpolation="nearest", cmap="Blues")
axes[0, 1].set_title("Tuned XGBoost — Confusion Matrix")
axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("Actual")
axes[0, 1].set_xticks([0, 1]); axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(["Normal", "Fraud"])
axes[0, 1].set_yticklabels(["Normal", "Fraud"])
for row in range(2):
    for col in range(2):
        axes[0, 1].text(col, row, f"{cm[row, col]:,}",
                        ha="center", va="center", fontweight="bold", fontsize=12,
                        color="white" if cm[row, col] > cm.max()/2 else "black")

fpr_b, tpr_b, _ = roc_curve(y_test, base_proba)
fpr_t, tpr_t, _ = roc_curve(y_test, tuned_proba)
axes[1, 0].plot(fpr_b, tpr_b, color="steelblue", linewidth=2,
                label=f"Before Tuning (AUC={base_auc:.3f})")
axes[1, 0].plot(fpr_t, tpr_t, color="crimson",   linewidth=2,
                label=f"After Tuning  (AUC={tuned_auc:.3f})")
axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1, 0].set_title("ROC Curve"); axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate"); axes[1, 0].legend()

axes[1, 1].barh(top_features[::-1], top_scores[::-1], color="steelblue", alpha=0.85)
axes[1, 1].set_title("Top 15 Feature Importances")
axes[1, 1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("notebooks/phase4_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Chart saved to notebooks/phase4_evaluation.png")

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(tuned_model, f)
print("✅ Tuned model saved to models/best_model.pkl")

print("\n" + "=" * 60)
print("  PHASE 4 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
  Model     : XGBoost (Tuned)
  Precision : {tuned_report['1']['precision']:.4f}
  Recall    : {tuned_report['1']['recall']:.4f}
  F1 Score  : {tuned_report['1']['f1-score']:.4f}
  AUC-ROC   : {tuned_auc:.4f}
  Best Params : {grid_search.best_params_}

  ➡  Next: Phase 5 — Guardian Demo UI
""")
