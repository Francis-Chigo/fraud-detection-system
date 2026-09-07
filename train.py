# ============================================================
# PHASE 3 - Model Building & Training
# NIBSS Nigerian Fraud Detection Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, os, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from xgboost import XGBClassifier

print("=" * 60)
print("  GUARDIAN - NIGERIAN PAYMENT FRAUD DETECTION")
print("  Phase 3: Model Training")
print("=" * 60)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("\n⏳ Loading processed data...")
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test  = pd.read_csv("data/processed/X_test.csv")
y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()
print(f"✅ Training set : {X_train.shape[0]:,} rows")
print(f"   Test set     : {X_test.shape[0]:,} rows")

# ── 2. DEFINE MODELS ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost"            : XGBClassifier(n_estimators=100, random_state=42,
                                          eval_metric="logloss", verbosity=0)
}
results = {}

# ── 3. TRAIN & EVALUATE ───────────────────────────────────────
for name, model in models.items():
    print(f"\n── Training {name} ──────────────────────────────────")
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)
    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "precision": report["1"]["precision"],
        "recall"   : report["1"]["recall"],
        "f1"       : report["1"]["f1-score"],
        "auc"      : auc, "cm": cm, "time": duration
    }
    print(f"✅ Done in {duration:.1f}s")
    print(f"   Precision : {report['1']['precision']:.4f}")
    print(f"   Recall    : {report['1']['recall']:.4f}")
    print(f"   F1 Score  : {report['1']['f1-score']:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")

# ── 4. COMPARISON TABLE ───────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Time':>8}")
print("-" * 75)
for name, r in results.items():
    print(f"{name:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} "
          f"{r['f1']:>10.4f} {r['auc']:>10.4f} {r['time']:>7.1f}s")

# ── 5. VISUALIZATIONS ─────────────────────────────────────────
print("\n⏳ Generating charts...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Guardian — Model Comparison (NIBSS Nigerian Dataset)",
             fontsize=14, fontweight="bold")
colors = ["steelblue", "seagreen", "crimson"]
names  = list(results.keys())

for i, (name, r) in enumerate(results.items()):
    cm = r["cm"]
    axes[0, i].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0, i].set_title(f"{name}\nConfusion Matrix", fontsize=10)
    axes[0, i].set_xlabel("Predicted"); axes[0, i].set_ylabel("Actual")
    axes[0, i].set_xticks([0, 1]); axes[0, i].set_yticks([0, 1])
    axes[0, i].set_xticklabels(["Normal", "Fraud"])
    axes[0, i].set_yticklabels(["Normal", "Fraud"])
    for row in range(2):
        for col in range(2):
            axes[0, i].text(col, row, f"{cm[row, col]:,}",
                            ha="center", va="center", fontweight="bold", fontsize=10,
                            color="white" if cm[row, col] > cm.max()/2 else "black")

metrics = ["precision", "recall", "f1", "auc"]
x = np.arange(len(metrics))
width = 0.25
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    axes[1, 0].bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
axes[1, 0].set_title("Metrics Comparison")
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels(["Precision", "Recall", "F1", "AUC"])
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_ylabel("Score")

for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    axes[1, 1].plot(fpr, tpr, color=colors[i],
                    label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1, 1].set_title("ROC Curves")
axes[1, 1].set_xlabel("False Positive Rate")
axes[1, 1].set_ylabel("True Positive Rate")
axes[1, 1].legend(fontsize=8)

times = [r["time"] for r in results.values()]
axes[1, 2].bar(names, times, color=colors, alpha=0.85)
axes[1, 2].set_title("Training Time (seconds)")
axes[1, 2].set_ylabel("Seconds")
axes[1, 2].set_xticklabels(names, rotation=15, ha="right", fontsize=9)
for i, v in enumerate(times):
    axes[1, 2].text(i, v + 0.5, f"{v:.1f}s", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("notebooks/phase3_models.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Chart saved to notebooks/phase3_models.png")

# ── 6. SAVE BEST MODEL ────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["f1"])
best_model = results[best_name]["model"]
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best model saved: {best_name}")

print("\n" + "=" * 60)
print("  PHASE 3 COMPLETE — SUMMARY")
print("=" * 60)
best = results[best_name]
print(f"""
  Best Model : {best_name}
  Precision  : {best['precision']:.4f}
  Recall     : {best['recall']:.4f}
  F1 Score   : {best['f1']:.4f}
  AUC-ROC    : {best['auc']:.4f}

  ➡  Next: Phase 4 — Model Evaluation & Tuning
""")
