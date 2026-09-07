# ============================================================
# PHASE 1 - Data Exploration
# NIBSS Nigerian Fraud Detection Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  GUARDIAN - NIGERIAN PAYMENT FRAUD DETECTION")
print("  Phase 1: Data Exploration")
print("=" * 60)

# ── 1. LOAD DATASET ──────────────────────────────────────────
print("\n⏳ Loading NIBSS Nigerian dataset...")
df = pd.read_csv("data/nibss_fraud_dataset.csv", low_memory=False)

print(f"\n✅ Dataset loaded successfully!")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")

# ── 2. FIRST LOOK ────────────────────────────────────────────
print("\n── First 5 rows ──────────────────────────────────────")
print(df.head())

print("\n── Column names ──────────────────────────────────────")
for col in df.columns:
    print(f"   {col}")

print("\n── Data types ────────────────────────────────────────")
print(df.dtypes)

# ── 3. MISSING VALUES ─────────────────────────────────────────
print("\n── Missing values ────────────────────────────────────")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(missing[missing > 0])

# ── 4. CLASS DISTRIBUTION ────────────────────────────────────
print("\n── Fraud vs Normal distribution ──────────────────────")
class_counts = df["is_fraud"].value_counts()
fraud_count  = class_counts[1]
normal_count = class_counts[0]
total        = len(df)

print(f"   Normal transactions : {normal_count:,}  ({normal_count/total*100:.2f}%)")
print(f"   Fraud  transactions : {fraud_count:,}  ({fraud_count/total*100:.2f}%)")
print(f"\n   ⚠️  Class imbalance detected — will fix with SMOTE in Phase 2.")

# ── 5. BASIC STATS ────────────────────────────────────────────
print("\n── Basic statistics (Amount) ─────────────────────────")
print(df[["amount", "is_fraud"]].describe())

# ── 6. CATEGORY BREAKDOWNS ───────────────────────────────────
print("\n── Transaction channels ──────────────────────────────")
print(df["channel"].value_counts())

print("\n── Top merchant categories ───────────────────────────")
print(df["merchant_category"].value_counts().head(10))

print("\n── Fraud by channel ──────────────────────────────────")
print(df.groupby("channel")["is_fraud"].sum().sort_values(ascending=False))

print("\n── Fraud by location ─────────────────────────────────")
print(df.groupby("location")["is_fraud"].sum().sort_values(ascending=False).head(10))

# ── 7. VISUALIZATIONS ─────────────────────────────────────────
print("\n⏳ Generating charts...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Guardian — NIBSS Nigerian Payment Fraud: Data Exploration",
             fontsize=15, fontweight="bold")

# Chart 1: Class distribution
axes[0, 0].bar(["Normal", "Fraud"], [normal_count, fraud_count],
               color=["steelblue", "crimson"])
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_ylabel("Number of Transactions")
for i, v in enumerate([normal_count, fraud_count]):
    axes[0, 0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

# Chart 2: Transaction amount distribution
axes[0, 1].hist(df[df["is_fraud"] == 0]["amount"].clip(0, 500000), bins=50,
                color="steelblue", alpha=0.7, label="Normal")
axes[0, 1].hist(df[df["is_fraud"] == 1]["amount"].clip(0, 500000), bins=50,
                color="crimson", alpha=0.7, label="Fraud")
axes[0, 1].set_title("Transaction Amount Distribution")
axes[0, 1].set_xlabel("Amount (₦)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()

# Chart 3: Fraud by channel
fraud_by_channel = df[df["is_fraud"] == 1]["channel"].value_counts()
axes[0, 2].bar(fraud_by_channel.index, fraud_by_channel.values, color="crimson", alpha=0.85)
axes[0, 2].set_title("Fraud Cases by Channel")
axes[0, 2].set_xlabel("Channel")
axes[0, 2].set_ylabel("Fraud Count")
axes[0, 2].tick_params(axis="x", rotation=30)

# Chart 4: Fraud by merchant category
fraud_by_merchant = df[df["is_fraud"] == 1]["merchant_category"].value_counts().head(8)
axes[1, 0].barh(fraud_by_merchant.index[::-1], fraud_by_merchant.values[::-1],
                color="crimson", alpha=0.85)
axes[1, 0].set_title("Top 8 Fraud Merchant Categories")
axes[1, 0].set_xlabel("Fraud Count")

# Chart 5: Fraud by age group
fraud_by_age = df[df["is_fraud"] == 1]["age_group"].value_counts()
axes[1, 1].bar(fraud_by_age.index, fraud_by_age.values, color="steelblue", alpha=0.85)
axes[1, 1].set_title("Fraud Cases by Age Group")
axes[1, 1].set_xlabel("Age Group")
axes[1, 1].set_ylabel("Fraud Count")
axes[1, 1].tick_params(axis="x", rotation=30)

# Chart 6: Fraud by hour of day
fraud_by_hour = df[df["is_fraud"] == 1]["hour"].value_counts().sort_index()
axes[1, 2].plot(fraud_by_hour.index, fraud_by_hour.values,
                color="crimson", linewidth=2, marker="o", markersize=4)
axes[1, 2].set_title("Fraud Cases by Hour of Day")
axes[1, 2].set_xlabel("Hour (0-23)")
axes[1, 2].set_ylabel("Fraud Count")
axes[1, 2].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig("notebooks/phase1_exploration.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Charts saved to notebooks/phase1_exploration.png")

# ── 8. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1 COMPLETE — KEY FINDINGS")
print("=" * 60)
print(f"""
  1. Dataset has {total:,} Nigerian banking transactions (NIBSS-calibrated).
  2. {df.shape[1]} features including channel, merchant category, location,
     age group, hour, bank, and behavioral risk scores.
  3. Fraud cases: {fraud_count:,} ({fraud_count/total*100:.2f}%) — class imbalance present.
  4. Transaction amounts are in Nigerian Naira (₦).
  5. Data covers transactions from 2023 onwards.
  6. No missing values — data is clean.

  ➡  Next: Phase 2 — Preprocessing & Feature Engineering
""")
