#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# 🏆 Customer Churn Prediction — Ridge Stacking Pipeline
## Kaggle Playground Series S6E3

This notebook is a **simple yet powerful** Ridge Stacking pipeline
designed for the customer churn prediction competition.

### 📋 Contents
1. **Data Loading & EDA** — Understanding the data with visualizations
2. **Feature Engineering** — Smart feature derivation
3. **OOF Pool** — Diverse predictions from XGBoost, LightGBM, CatBoost
4. **Ridge Stacking** — Combining OOFs for final prediction
5. **Submission** — Generating the submission file

### ⏱️ Estimated Runtime: ~45-60 min (CPU)
"""

# ============================================================================
# 📦 1. LIBRARIES & SETTINGS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import time
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

print("=" * 70)
print("🏆 CUSTOMER CHURN — RIDGE STACKING PIPELINE")
print("=" * 70)

# ============================================================================
# 📥 2. DATA LOADING
# ============================================================================
"""
## 📥 Data Loading
Competition data and optionally the original Telco dataset are loaded.
"""
start_time = time.time()

if os.path.exists('/kaggle/input/competitions/playground-series-s6e3/train.csv'):
    DATA_DIR = '/kaggle/input/competitions/playground-series-s6e3'
else:
    DATA_DIR = 'data'

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

# Target encoding
train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1})

y = train['Churn'].values
test_ids = test['id'].values
n_train = len(train)
n_test = len(test)

print(f"📊 Train: {train.shape} | Test: {test.shape}")
print(f"🎯 Churn rate: {y.mean():.4f} ({y.sum():,} / {len(y):,})")
print(f"⏱️  Data loading: {time.time()-start_time:.1f}s")

# ============================================================================
# 📊 3. EDA — EXPLORATORY DATA ANALYSIS
# ============================================================================
"""
## 📊 Exploratory Data Analysis (EDA)
Visualizing the dataset structure and churn distribution.
"""

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('📊 Customer Churn — EDA', fontsize=16, fontweight='bold')

# 1. Churn distribution
ax = axes[0, 0]
churn_counts = train['Churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax.bar(['Stayed (0)', 'Churned (1)'], churn_counts.values, color=colors, edgecolor='white')
for i, v in enumerate(churn_counts.values):
    ax.text(i, v + 500, f'{v:,}\n({v/len(train)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Churn Distribution', fontweight='bold')
ax.set_ylabel('Number of Customers')

# 2. Tenure distribution
ax = axes[0, 1]
ax.hist(train[train['Churn']==0]['tenure'], bins=50, alpha=0.6, label='Stayed', color='#2ecc71')
ax.hist(train[train['Churn']==1]['tenure'], bins=50, alpha=0.6, label='Churned', color='#e74c3c')
ax.set_title('Tenure Distribution (by Churn)', fontweight='bold')
ax.set_xlabel('Tenure (Months)')
ax.legend()

# 3. MonthlyCharges distribution
ax = axes[0, 2]
ax.hist(train[train['Churn']==0]['MonthlyCharges'], bins=50, alpha=0.6, label='Stayed', color='#2ecc71')
ax.hist(train[train['Churn']==1]['MonthlyCharges'], bins=50, alpha=0.6, label='Churned', color='#e74c3c')
ax.set_title('Monthly Charges (by Churn)', fontweight='bold')
ax.set_xlabel('Monthly Charge ($)')
ax.legend()

# 4. Contract type and churn
ax = axes[1, 0]
if 'Contract' in train.columns:
    ct = train.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
    bars = ax.bar(ct.index, ct.values, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white')
    for bar, v in zip(bars, ct.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    ax.set_title('Contract Type → Churn Rate', fontweight='bold')
    ax.set_ylabel('Churn Rate')

# 5. Internet service and churn
ax = axes[1, 1]
if 'InternetService' in train.columns:
    it = train.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)
    bars = ax.bar(it.index, it.values, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='white')
    for bar, v in zip(bars, it.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    ax.set_title('Internet Service → Churn Rate', fontweight='bold')
    ax.set_ylabel('Churn Rate')

# 6. Payment method and churn
ax = axes[1, 2]
if 'PaymentMethod' in train.columns:
    pm = train.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False)
    bars = ax.barh(pm.index, pm.values, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='white')
    for bar, v in zip(bars, pm.values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center', fontsize=10)
    ax.set_title('Payment Method → Churn Rate', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA charts generated")

# ============================================================================
# 🔧 4. FEATURE ENGINEERING
# ============================================================================
"""
## 🔧 Feature Engineering
Deriving new features based on domain knowledge.
"""
print("\n🔧 Feature Engineering starting...")

# Column types
drop_cols = ['id', 'Churn', 'customerID']
cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in train.select_dtypes(exclude='object').columns if c not in drop_cols]

# TotalCharges cleanup
for df in [train, test]:
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']

def engineer_features(df):
    """Domain-driven feature engineering"""
    out = df.copy()

    # --- Service-based ---
    if services[0] in out.columns and out[services[0]].dtype == 'object':
        out['Total_Services'] = (out[services] == 'Yes').sum(axis=1)
        out['Total_No_Services'] = (out[services] == 'No').sum(axis=1)
    else:
        out['Total_Services'] = 0
        out['Total_No_Services'] = 0

    # --- Financial ---
    out['Avg_Monthly_Spend'] = out['TotalCharges'] / (out['tenure'] + 1)
    out['Charge_Per_Service'] = out['MonthlyCharges'] / (out['Total_Services'] + 1)
    out['Contract_Value'] = out['tenure'] * out['MonthlyCharges'] - out['TotalCharges']
    out['Monthly_to_Total'] = out['MonthlyCharges'] / (out['TotalCharges'] + 1)
    out['Tenure_x_Monthly'] = out['tenure'] * out['MonthlyCharges']

    # --- Tenure ---
    out['Is_New_Customer'] = (out['tenure'] <= 6).astype(int)
    out['Is_VIP_Customer'] = (out['tenure'] >= 48).astype(int)
    out['Tenure_Squared'] = out['tenure'] ** 2
    out['Tenure_Log'] = np.log1p(out['tenure'])

    # --- Risk flags ---
    out['Is_Overpaying'] = (out['MonthlyCharges'] > out['Avg_Monthly_Spend']).astype(int)

    if 'OnlineSecurity' in df.columns and 'TechSupport' in df.columns:
        out['Has_No_Support'] = ((df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)

    if 'InternetService' in df.columns:
        out['Fiber_Optic'] = (df['InternetService'] == 'Fiber optic').astype(int)

    if 'Contract' in df.columns:
        out['Month_to_Month'] = (df['Contract'] == 'Month-to-month').astype(int)

    if 'PaymentMethod' in df.columns:
        out['Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

    # --- Interaction features ---
    if 'Is_New_Customer' in out.columns and 'Month_to_Month' in out.columns:
        out['New_MonthToMonth'] = out['Is_New_Customer'] * out['Month_to_Month']

    if 'Fiber_Optic' in out.columns and 'Has_No_Support' in out.columns:
        out['Fiber_NoSupport'] = out['Fiber_Optic'] * out['Has_No_Support']

    if 'Is_New_Customer' in out.columns:
        out['New_HighCharge'] = (out['Is_New_Customer'] * (out['MonthlyCharges'] > 70)).astype(int)

    return out

train_eng = engineer_features(train)
test_eng = engineer_features(test)

# Label encoding
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_eng[col], test_eng[col]]).astype(str)
    le.fit(combined)
    train_eng[col] = le.transform(train_eng[col].astype(str))
    test_eng[col] = le.transform(test_eng[col].astype(str))

# Feature matrix
feature_cols = [c for c in train_eng.columns if c not in drop_cols]
X = train_eng[feature_cols].values.astype(np.float32)
X_test_final = test_eng[feature_cols].values.astype(np.float32)

# NaN cleanup
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=1e6, neginf=-1e6)

print(f"   ✅ {X.shape[1]} features created")
print(f"   📋 Examples: {feature_cols[:10]}...")

# ============================================================================
# 📊 5. FEATURE IMPORTANCE VISUALIZATION
# ============================================================================
"""
## 📊 Feature Importance
Visualizing the most important features using a quick LightGBM model.
"""
print("\n📊 Calculating feature importance...")

quick_lgb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                                random_state=SEED, verbose=-1)
quick_lgb.fit(X, y)
importances = quick_lgb.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})\
         .sort_values('Importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1],
               color=plt.cm.viridis(np.linspace(0.3, 0.9, 20)), edgecolor='white')
ax.set_title('🏅 Top 20 Feature Importance (LightGBM)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance chart generated")

# ============================================================================
# 🏊 6. OOF POOL GENERATION
# ============================================================================
"""
## 🏊 OOF Pool (Out-of-Fold Predictions)
Generating OOF predictions for each model × configuration combo.
These predictions will be combined via Ridge Stacking.
"""
print(f"\n{'='*70}")
print("🏊 GENERATING OOF POOL")
print(f"{'='*70}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLD_INDICES = list(skf.split(X, y))

oof_pool = {}
test_pool = {}
best_iters = {}

def generate_oof(name, X_tr, y_tr, X_te, model_fn):
    """Generates OOF predictions."""
    oof_pred = np.zeros(n_train)
    test_pred = np.zeros(n_test)
    iterations = []

    for fold, (tr_idx, val_idx) in enumerate(FOLD_INDICES):
        model = model_fn()

        if 'XGB' in name:
            model.fit(X_tr[tr_idx], y_tr[tr_idx],
                      eval_set=[(X_tr[val_idx], y_tr[val_idx])], verbose=False)
            iterations.append(model.best_iteration if hasattr(model, 'best_iteration')
                            else model.get_booster().best_iteration)
        elif 'LGBM' in name:
            model.fit(X_tr[tr_idx], y_tr[tr_idx],
                      eval_set=[(X_tr[val_idx], y_tr[val_idx])])
            iterations.append(model.best_iteration_)
        elif 'CAT' in name:
            model.fit(X_tr[tr_idx], y_tr[tr_idx],
                      eval_set=[(X_tr[val_idx], y_tr[val_idx])], verbose=False)
            iterations.append(model.get_best_iteration())

        oof_pred[val_idx] = model.predict_proba(X_tr[val_idx])[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS

    auc = roc_auc_score(y_tr, oof_pred)
    oof_pool[name] = oof_pred
    test_pool[name] = test_pred
    if iterations:
        best_iters[name] = int(np.mean(iterations))
    print(f"   🎯 {name:<30s} AUC: {auc:.5f}  (avg iter: {int(np.mean(iterations)) if iterations else 'N/A'})")
    return auc

# ---- Model Configurations ----
"""
### Model Diversity
Different hyperparameters → different errors → stronger ensemble
"""

# XGBoost
print("\n--- XGBoost ---")
xgb_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 6,
     'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
     'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'n_estimators': 2000, 'learning_rate': 0.05, 'max_depth': 4,
     'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 10,
     'reg_alpha': 1.0, 'reg_lambda': 5.0},
]
for i, cfg in enumerate(xgb_configs):
    def make_xgb(c=cfg):
        return xgb.XGBClassifier(**c, tree_method='hist', random_state=SEED,
                                  eval_metric='auc', early_stopping_rounds=100, verbosity=0)
    generate_oof(f'XGB_c{i}', X, y, X_test_final, make_xgb)

# LightGBM
print("\n--- LightGBM ---")
lgbm_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 7,
     'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_samples': 20,
     'num_leaves': 63, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'n_estimators': 2000, 'learning_rate': 0.05, 'max_depth': 5,
     'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_samples': 50,
     'num_leaves': 31, 'reg_alpha': 1.0, 'reg_lambda': 5.0},
]
for i, cfg in enumerate(lgbm_configs):
    def make_lgbm(c=cfg):
        return lgb.LGBMClassifier(**c, random_state=SEED, verbose=-1,
                                   early_stopping_rounds=100)
    generate_oof(f'LGBM_c{i}', X, y, X_test_final, make_lgbm)

# CatBoost
print("\n--- CatBoost ---")
cat_configs = [
    {'iterations': 2000, 'learning_rate': 0.03, 'depth': 7,
     'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5},
    {'iterations': 2000, 'learning_rate': 0.05, 'depth': 5,
     'l2_leaf_reg': 5.0, 'bagging_temperature': 1.0},
]
for i, cfg in enumerate(cat_configs):
    def make_cat(c=cfg):
        return CatBoostClassifier(**c, random_seed=SEED, verbose=False,
                                   eval_metric='AUC', early_stopping_rounds=100)
    generate_oof(f'CAT_c{i}', X, y, X_test_final, make_cat)

print(f"\n📊 Total OOFs: {len(oof_pool)}")

# ============================================================================
# 📊 7. OOF CORRELATION MATRIX
# ============================================================================
"""
## 📊 OOF Correlation Matrix
Visualizing inter-model correlation.
Low correlation = high diversity = stronger ensemble.
"""
oof_names = list(oof_pool.keys())
oof_matrix = np.column_stack([oof_pool[n] for n in oof_names])
test_matrix = np.column_stack([test_pool[n] for n in oof_names])

corr_df = pd.DataFrame(oof_matrix, columns=oof_names).corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.3f', cmap='RdYlGn_r',
            center=0.95, vmin=0.9, vmax=1.0, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('🔗 OOF Correlation Matrix\n(Lower = More Diversity = Better)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('oof_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation matrix generated")

# ============================================================================
# 🏔️ 8. RIDGE STACKING
# ============================================================================
"""
## 🏔️ Ridge Regression Stacking
Combining OOF predictions with Ridge Regression.
Ridge is simple, stable, and resistant to overfitting — the choice of Kaggle champions.
"""
print(f"\n{'='*70}")
print("🏔️ RIDGE STACKING")
print(f"{'='*70}")

# Alpha grid search
best_ridge_auc = 0.0
best_alpha = 1.0
results = []

for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    oof_ridge = np.zeros(n_train)
    for tr_idx, val_idx in FOLD_INDICES:
        ridge = Ridge(alpha=alpha)
        ridge.fit(oof_matrix[tr_idx], y[tr_idx])
        oof_ridge[val_idx] = ridge.predict(oof_matrix[val_idx])

    auc = roc_auc_score(y, oof_ridge)
    results.append({'alpha': alpha, 'AUC': auc})
    if auc > best_ridge_auc:
        best_ridge_auc = auc
        best_alpha = alpha

print(f"\n   ✅ Best alpha: {best_alpha}")
print(f"   🎯 Ridge OOF AUC: {best_ridge_auc:.5f}")

# Alpha vs AUC chart
fig, ax = plt.subplots(figsize=(8, 5))
res_df = pd.DataFrame(results)
ax.plot(res_df['alpha'], res_df['AUC'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
ax.axvline(x=best_alpha, color='#2ecc71', linestyle='--', label=f'Best α={best_alpha}')
ax.set_xscale('log')
ax.set_title('🏔️ Ridge Alpha vs AUC', fontsize=13, fontweight='bold')
ax.set_xlabel('Alpha (log scale)')
ax.set_ylabel('OOF AUC')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ridge_alpha.png', dpi=150, bbox_inches='tight')
plt.show()

# Individual vs Ridge comparison
"""
### 📊 Model Comparison
Individual AUC of each model vs Ridge Stacking
"""
individual_aucs = {n: roc_auc_score(y, oof_pool[n]) for n in oof_names}
simple_avg_auc = roc_auc_score(y, oof_matrix.mean(axis=1))

fig, ax = plt.subplots(figsize=(10, 5))
names_sorted = sorted(individual_aucs, key=individual_aucs.get, reverse=True)
aucs_sorted = [individual_aucs[n] for n in names_sorted]

bars = ax.bar(range(len(names_sorted)), aucs_sorted, color='#3498db', alpha=0.7, label='Individual')
ax.axhline(y=simple_avg_auc, color='#f39c12', linewidth=2, linestyle='--', label=f'Average ({simple_avg_auc:.5f})')
ax.axhline(y=best_ridge_auc, color='#e74c3c', linewidth=2.5, label=f'Ridge ({best_ridge_auc:.5f})')
ax.set_xticks(range(len(names_sorted)))
ax.set_xticklabels(names_sorted, rotation=45, ha='right')
ax.set_ylabel('AUC')
ax.set_title('🏅 Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(min(aucs_sorted) - 0.001, best_ridge_auc + 0.001)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 📋 9. FINAL REPORT
# ============================================================================
"""
## 📋 Final Results Report
"""
# Final Ridge prediction
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(oof_matrix, y)
final_pred = np.clip(final_ridge.predict(test_matrix), 0, 1)

# Ridge weights
fig, ax = plt.subplots(figsize=(8, 5))
weights = final_ridge.coef_
colors = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]
ax.barh(oof_names, weights, color=colors, edgecolor='white')
ax.set_title('🏔️ Ridge Model Weights', fontsize=13, fontweight='bold')
ax.set_xlabel('Weight')
ax.axvline(x=0, color='gray', linewidth=0.5)
plt.tight_layout()
plt.savefig('ridge_weights.png', dpi=150, bbox_inches='tight')
plt.show()

elapsed = time.time() - start_time

print(f"\n{'='*70}")
print("📋 FINAL RESULTS REPORT")
print(f"{'='*70}")
print(f"\n   Total models          : {len(oof_pool)}")
print(f"   Best individual AUC   : {max(individual_aucs.values()):.5f} ({max(individual_aucs, key=individual_aucs.get)})")
print(f"   Simple Average AUC    : {simple_avg_auc:.5f}")
print(f"   🏔️ Ridge Stacking AUC : {best_ridge_auc:.5f}")
print(f"   Ridge alpha           : {best_alpha}")
print(f"   ⏱️ Total time          : {elapsed/60:.1f} minutes")
print(f"{'='*70}")

# ============================================================================
# 💾 10. SUBMISSION
# ============================================================================
"""
## 💾 Submission
Generating the submission file for Kaggle.
"""
submission = pd.DataFrame({
    'id': test_ids,
    'Churn': final_pred
})

output_path = '/kaggle/working/submission.csv' if os.path.exists('/kaggle/working') else 'submission_simple.csv'
submission.to_csv(output_path, index=False)
print(f"\n✅ Submission file: {output_path}")
print(f"   📊 Prediction range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
print(f"   📊 Prediction mean: {final_pred.mean():.4f}")

print(f"\n🏆 Pipeline completed! ({elapsed/60:.1f} min)")
