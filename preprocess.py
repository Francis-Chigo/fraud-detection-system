# ============================================================
# PHASE 2 - Data Preprocessing & Feature Engineering
# NIBSS Nigerian Fraud Detection Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

print("=" * 60)
print("  GUARDIAN - NIGERIAN PAYMENT FRAUD DETECTION")
print("  Phase 2: Data Preprocessing")
print("=" * 60)

# ── 1. LOAD DATASET ──────────────────────────────────────────
print("\n⏳ Loading NIBSS Nigerian dataset...")
df = pd.read_csv("data/nibss_fraud_dataset.csv", low_memory=False)
print(f"✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# ── 2. DROP IRRELEVANT COLUMNS ────────────────────────────────
print("\n⏳ Dropping non-predictive columns...")
drop_cols = ["transaction_id", "customer_id", "timestamp", "fraud_technique"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"✅ Dropped: {drop_cols}")

# ── 3. ENCODE CATEGORICAL COLUMNS ────────────────────────────
print("\n⏳ Encoding categorical columns...")
cat_cols = ["channel", "merchant_category", "bank", "location",
            "age_group", "day_of_week"]
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
print(f"✅ Encoded: {cat_cols}")

# ── 4. SCALE NUMERIC COLUMNS ─────────────────────────────────
print("\n⏳ Scaling numeric columns...")
scale_cols = ["amount", "hour", "month", "tx_count_24h",
              "amount_sum_24h", "amount_mean_7d", "amount_std_7d",
              "tx_count_total", "amount_mean_total", "amount_std_total",
              "velocity_score", "merchant_risk_score", "composite_risk",
              "amount_log", "amount_rounded", "amount_vs_mean_ratio",
              "channel_diversity", "location_diversity", "online_channel_ratio"]
scale_cols = [c for c in scale_cols if c in df.columns]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print(f"✅ Scaled {len(scale_cols)} numeric columns")

# ── 5. SPLIT FEATURES AND TARGET ─────────────────────────────
print("\n⏳ Splitting features and target...")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
print(f"✅ Features (X) shape : {X.shape}")
print(f"   Target  (y) shape : {y.shape}")
print(f"   Fraud cases before SMOTE: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")

# ── 6. TRAIN / TEST SPLIT ─────────────────────────────────────
print("\n⏳ Splitting into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training set : {X_train.shape[0]:,} rows")
print(f"   Test set     : {X_test.shape[0]:,} rows")

# ── 7. APPLY SMOTE ────────────────────────────────────────────
print("\n⏳ Applying SMOTE to fix class imbalance...")
print("   (This may take 2-3 minutes on 1M rows...)")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"✅ SMOTE applied successfully!")
print(f"   Training set before SMOTE : {len(y_train):,} rows")
print(f"   Training set after SMOTE  : {len(y_train_sm):,} rows")
print(f"   Fraud cases after SMOTE   : {y_train_sm.sum():,} ({y_train_sm.sum()/len(y_train_sm)*100:.2f}%)")

# ── 8. VISUALIZE BEFORE vs AFTER SMOTE ───────────────────────
print("\n⏳ Generating charts...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Guardian — Class Balance Before vs After SMOTE (NIBSS Dataset)",
             fontsize=13, fontweight="bold")

before_counts = y_train.value_counts()
axes[0].bar(["Normal", "Fraud"], [before_counts[0], before_counts[1]],
            color=["steelblue", "crimson"])
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Number of Transactions")
for i, v in enumerate([before_counts[0], before_counts[1]]):
    axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

after_counts = pd.Series(y_train_sm).value_counts()
axes[1].bar(["Normal", "Fraud"], [after_counts[0], after_counts[1]],
            color=["steelblue", "crimson"])
axes[1].set_title("After SMOTE")
axes[1].set_ylabel("Number of Transactions")
for i, v in enumerate([after_counts[0], after_counts[1]]):
    axes[1].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("notebooks/phase2_smote.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Chart saved to notebooks/phase2_smote.png")

# ── 9. SAVE PROCESSED DATA ────────────────────────────────────
print("\n⏳ Saving processed data...")
os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_train_sm).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(y_train_sm).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

print("✅ Processed data saved to data/processed/")

print("\n" + "=" * 60)
print("  PHASE 2 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
  1. Dropped non-predictive columns (IDs, timestamps).
  2. Encoded categorical features (channel, bank, location, etc.)
  3. Scaled all numeric features with StandardScaler.
  4. Split data 80% training / 20% testing.
  5. SMOTE applied — classes balanced in training set.
  6. Processed data saved to data/processed/

  ➡  Next: Phase 3 — Model Building & Training
""")
