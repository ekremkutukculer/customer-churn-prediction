"""
🚀 CUSTOMER CHURN PREDİCTION — ULTIMATE KAGGLE PIPELINE v4
============================================================
Playground Series S6E3
Tüm iyileştirmeler tek bir script'te:
  1. Kapsamlı Feature Engineering
  2. StratifiedKFold (10-Fold)
  3. Multi-Seed Averaging (5 seed)
  4. Optuna Hyperparameter Tuning
  5. Weighted Ensemble (XGBoost + LightGBM + CatBoost)
  6. Target Encoding

Kaggle Notebook kullanımı:
  - GPU accelerator açın
  - Yarışma datasını ekleyin
  - Bu scripti çalıştırın
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║  YAPILANDIRMA (CONFIGURATION)                               ║
# ╚══════════════════════════════════════════════════════════════╝
N_FOLDS        = 10
SEEDS          = [42, 123, 456, 789, 2024]
OPTUNA_TRIALS  = 100          # Tuning deneme sayısı (daha fazla = daha iyi ama yavaş)
USE_OPTUNA     = True        # False yaparsanız varsayılan parametreleri kullanır
USE_GPU        = True        # Kaggle GPU accelerator açıksa True

# Kaggle mı yoksa lokal mi?
if os.path.exists('/kaggle/input/playground-series-s6e3/train.csv'):
    DATA_DIR = '/kaggle/input/playground-series-s6e3'
    OUTPUT_DIR = '/kaggle/working'
else:
    DATA_DIR = 'data'
    OUTPUT_DIR = '.'

print("🚀 ULTIMATE CHURN PIPELINE v4 BAŞLIYOR 🚀")
print("=" * 75)
print(f"📁 Veri dizini : {DATA_DIR}")
print(f"📊 Fold sayısı : {N_FOLDS}")
print(f"🌱 Seed sayısı : {len(SEEDS)}")
print(f"🎯 Optuna      : {'Açık' if USE_OPTUNA else 'Kapalı'} ({OPTUNA_TRIALS} deneme)")
print(f"🖥️  GPU          : {'Açık' if USE_GPU else 'Kapalı'}")
print("=" * 75)

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. VERİ YÜKLEME (DATA LOADING)                             ║
# ╚══════════════════════════════════════════════════════════════╝
t0 = time.time()
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')
test_ids = test_df['id'].copy()

print(f"\n📥 Veri yüklendi: Train={train_df.shape}, Test={test_df.shape}  ({time.time()-t0:.1f}s)")

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)             ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n🔧 Feature Engineering başlıyor...")

hizmetler = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
             'TechSupport', 'StreamingTV', 'StreamingMovies']

for df in [train_df, test_df]:
    # --- Hizmet bazlı özellikler ---
    df['Total_Services']    = (df[hizmetler] == 'Yes').sum(axis=1)
    df['Total_No_Services'] = (df[hizmetler] == 'No').sum(axis=1)
    df['Has_Any_Service']   = (df['Total_Services'] > 0).astype(int)

    # --- Finansal özellikler ---
    df['Avg_Monthly_Spend']    = df['TotalCharges'] / (df['tenure'] + 1)
    df['Charge_Per_Service']   = df['MonthlyCharges'] / (df['Total_Services'] + 1)
    df['Tenure_x_Monthly']     = df['tenure'] * df['MonthlyCharges']
    df['Contract_Value']       = df['tenure'] * df['MonthlyCharges'] - df['TotalCharges']
    df['Monthly_to_Total']     = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['Is_Overpaying']        = (df['MonthlyCharges'] > df['Avg_Monthly_Spend']).astype(int)
    df['Monthly_Charge_Diff']  = df['MonthlyCharges'] - df['Avg_Monthly_Spend']

    # --- Tenure bazlı özellikler ---
    df['Is_New_Customer']  = (df['tenure'] <= 6).astype(int)
    df['Is_VIP_Customer']  = (df['tenure'] >= 48).astype(int)
    df['Tenure_Squared']   = df['tenure'] ** 2
    df['Tenure_Log']       = np.log1p(df['tenure'])
    df['Tenure_Group']     = pd.cut(df['tenure'], bins=[-1, 6, 12, 24, 48, 100],
                                     labels=[0, 1, 2, 3, 4]).astype(int)

    # --- Bağlılık / Risk göstergeleri ---
    df['Has_No_Support']     = ((df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)
    df['Has_Streaming']      = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
    df['Has_Partner_Dep']    = ((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes')).astype(int)
    df['No_Internet']        = (df['InternetService'] == 'No').astype(int)
    df['Fiber_Optic']        = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Auto_Payment']       = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
    df['Electronic_Check']   = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['Month_to_Month']     = (df['Contract'] == 'Month-to-month').astype(int)
    df['Paperless']          = (df['PaperlessBilling'] == 'Yes').astype(int)

    # --- Etkileşim (Interaction) özellikleri ---
    df['New_And_MonthToMonth']     = df['Is_New_Customer'] * df['Month_to_Month']
    df['Fiber_NoSupport']          = df['Fiber_Optic'] * df['Has_No_Support']
    df['Senior_MonthToMonth']      = df['SeniorCitizen'] * df['Month_to_Month']
    df['Electronic_MonthToMonth']  = df['Electronic_Check'] * df['Month_to_Month']
    df['Senior_NoSupport']         = df['SeniorCitizen'] * df['Has_No_Support']
    df['New_Fiber']                = df['Is_New_Customer'] * df['Fiber_Optic']
    df['Charges_x_NoSupport']      = df['MonthlyCharges'] * df['Has_No_Support']
    df['New_HighCharge']           = df['Is_New_Customer'] * (df['MonthlyCharges'] > 70).astype(int)

print(f"   ✅ Toplam {train_df.shape[1]} sütun oluşturuldu.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  3. KODLAMA (ENCODING)                                      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n🏷️  Encoding başlıyor...")

train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})
y = train_df['Churn'].values

drop_cols = ['id', 'Churn']
X = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=['id'])

# Kategorik sütunları belirle
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

# Label Encoding (CatBoost için native kategorik de ekleyebiliriz ama tutarlılık için LE)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Target encoding ek özellikler (KFold-safe)
print("   🎯 Target Encoding (KFold-safe)...")
te_cols_source = ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines']
# LE zaten uygulandı, orijinal sütun adlarını kullanıyoruz
for col in te_cols_source:
    te_col_name = f'{col}_TE'
    X[te_col_name] = np.nan
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_te.split(X, y):
        means = y[tr_idx].groupby(X[col].iloc[tr_idx]).mean() if hasattr(y, 'groupby') else \
                pd.Series(y[tr_idx]).groupby(X[col].iloc[tr_idx].values).mean()
        X.iloc[val_idx, X.columns.get_loc(te_col_name)] = X[col].iloc[val_idx].map(means)
    # Test seti: tüm train üzerinden hesapla
    global_means = pd.Series(y).groupby(X[col].values).mean()
    X_test[te_col_name] = X_test[col].map(global_means)
    # NaN kaldıysa genel ortalama ile doldur
    overall_mean = y.mean()
    X[te_col_name].fillna(overall_mean, inplace=True)
    X_test[te_col_name].fillna(overall_mean, inplace=True)

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)
feature_names = X.columns.tolist()
X = X.values
X_test_vals = X_test.values

print(f"   ✅ {len(feature_names)} özellik hazır.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  4. OPTUNA HYPERPARAMETER TUNING                             ║
# ╚══════════════════════════════════════════════════════════════╝
# Varsayılan parametreler (Optuna kapalıysa veya fallback)
default_xgb_params = {
    'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 5,
    'gamma': 0.1,
}
default_lgbm_params = {
    'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 7,
    'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 20,
    'num_leaves': 63,
}
default_cat_params = {
    'iterations': 1500, 'learning_rate': 0.03, 'depth': 7,
    'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5,
    'random_strength': 0.5,
}

if USE_OPTUNA:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("\n🔍 OPTUNA Hyperparameter Tuning Başlıyor...")

    # --- XGBoost Tuning ---
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
        skf_opt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf_opt.split(X, y):
            xgb_kw = {'tree_method': 'hist', 'random_state': 42, 'eval_metric': 'auc',
                       'early_stopping_rounds': 100, 'verbosity': 0}
            if USE_GPU:
                xgb_kw['device'] = 'cuda'
            model = XGBClassifier(**params, **xgb_kw)
            model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            pred = model.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], pred))
        return np.mean(scores)

    print("   ⏳ XGBoost tuning...")
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    print(f"   ✅ XGBoost en iyi AUC: {xgb_study.best_value:.5f}")

    # --- LightGBM Tuning ---
    def lgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        }
        skf_opt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf_opt.split(X, y):
            model = LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])])
            pred = model.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], pred))
        return np.mean(scores)

    print("   ⏳ LightGBM tuning...")
    lgbm_study = optuna.create_study(direction='maximize')
    lgbm_study.optimize(lgbm_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_lgbm_params = lgbm_study.best_params
    print(f"   ✅ LightGBM en iyi AUC: {lgbm_study.best_value:.5f}")

    # --- CatBoost Tuning ---
    def cat_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 800, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        }
        skf_opt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf_opt.split(X, y):
            cat_kw = {'random_seed': 42, 'verbose': False, 'eval_metric': 'AUC',
                       'early_stopping_rounds': 100}
            if USE_GPU:
                cat_kw['task_type'] = 'GPU'
            model = CatBoostClassifier(**params, **cat_kw)
            model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            pred = model.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], pred))
        return np.mean(scores)

    print("   ⏳ CatBoost tuning...")
    cat_study = optuna.create_study(direction='maximize')
    cat_study.optimize(cat_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_cat_params = cat_study.best_params
    print(f"   ✅ CatBoost en iyi AUC: {cat_study.best_value:.5f}")

else:
    best_xgb_params  = default_xgb_params
    best_lgbm_params = default_lgbm_params
    best_cat_params  = default_cat_params
    print("\n⏭️  Optuna kapalı — varsayılan parametreler kullanılıyor.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  5. MULTI-SEED KFOLD EĞİTİMİ (TRAINING)                    ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 75)
print("👑 MULTI-SEED KFOLD EĞİTİMİ BAŞLIYOR 👑")
print("=" * 75)

def train_model_kfold(model_name, create_model_fn, X_tr, y_tr, X_te, n_folds, seeds):
    """Her seed için KFold eğitimi yapar, OOF ve test tahminlerinin ortalamasını döner."""
    all_oof   = np.zeros((len(seeds), len(y_tr)))
    all_test  = np.zeros((len(seeds), X_te.shape[0]))

    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_pred  = np.zeros(len(y_tr))
        test_pred = np.zeros(X_te.shape[0])

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            model = create_model_fn(seed)

            if model_name == 'XGBoost':
                model.fit(X_tr[tr_idx], y_tr[tr_idx],
                          eval_set=[(X_tr[val_idx], y_tr[val_idx])], verbose=False)
            elif model_name == 'LightGBM':
                model.fit(X_tr[tr_idx], y_tr[tr_idx],
                          eval_set=[(X_tr[val_idx], y_tr[val_idx])])
            elif model_name == 'CatBoost':
                model.fit(X_tr[tr_idx], y_tr[tr_idx],
                          eval_set=[(X_tr[val_idx], y_tr[val_idx])], verbose=False)

            oof_pred[val_idx] = model.predict_proba(X_tr[val_idx])[:, 1]
            test_pred += model.predict_proba(X_te)[:, 1] / n_folds

        fold_auc = roc_auc_score(y_tr, oof_pred)
        print(f"   🌱 Seed {seed:>5d} → OOF AUC: {fold_auc:.5f}")
        all_oof[s_idx]  = oof_pred
        all_test[s_idx] = test_pred

    mean_oof  = all_oof.mean(axis=0)
    mean_test = all_test.mean(axis=0)
    mean_auc  = roc_auc_score(y_tr, mean_oof)
    print(f"   📊 {model_name} Ortalama OOF AUC: {mean_auc:.5f}")
    return mean_oof, mean_test, mean_auc

# --- XGBoost ---
print(f"\n⚡ XGBoost eğitiliyor ({len(SEEDS)} seed × {N_FOLDS} fold)...")
def create_xgb(seed):
    kw = {**best_xgb_params, 'tree_method': 'hist', 'random_state': seed,
          'eval_metric': 'auc', 'early_stopping_rounds': 100, 'verbosity': 0}
    if USE_GPU:
        kw['device'] = 'cuda'
    return XGBClassifier(**kw)

xgb_oof, xgb_test, xgb_auc = train_model_kfold('XGBoost', create_xgb, X, y, X_test_vals, N_FOLDS, SEEDS)

# --- LightGBM ---
print(f"\n⚡ LightGBM eğitiliyor ({len(SEEDS)} seed × {N_FOLDS} fold)...")
def create_lgbm(seed):
    return LGBMClassifier(**best_lgbm_params, random_state=seed, verbose=-1,
                           early_stopping_rounds=100)

lgbm_oof, lgbm_test, lgbm_auc = train_model_kfold('LightGBM', create_lgbm, X, y, X_test_vals, N_FOLDS, SEEDS)

# --- CatBoost ---
print(f"\n⚡ CatBoost eğitiliyor ({len(SEEDS)} seed × {N_FOLDS} fold)...")
def create_cat(seed):
    kw = {**best_cat_params, 'random_seed': seed, 'verbose': False,
          'eval_metric': 'AUC', 'early_stopping_rounds': 100}
    if USE_GPU:
        kw['task_type'] = 'GPU'
    return CatBoostClassifier(**kw)

cat_oof, cat_test, cat_auc = train_model_kfold('CatBoost', create_cat, X, y, X_test_vals, N_FOLDS, SEEDS)

# ╔══════════════════════════════════════════════════════════════╗
# ║  6. OPTİMAL AĞIRLIKLI ENSEMBLE                              ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 75)
print("🏆 WEIGHTED ENSEMBLE OPTİMİZASYONU 🏆")
print("=" * 75)

def neg_auc(weights):
    w = np.abs(weights)
    w = w / w.sum()
    blend = w[0] * xgb_oof + w[1] * lgbm_oof + w[2] * cat_oof
    return -roc_auc_score(y, blend)

# Başlangıç noktalarını çeşitlendirerek global optimum bulma
best_result = None
for init_w in [[1/3, 1/3, 1/3], [0.5, 0.3, 0.2], [0.2, 0.5, 0.3],
               [0.3, 0.2, 0.5], [0.4, 0.4, 0.2], [0.2, 0.4, 0.4]]:
    result = minimize(neg_auc, x0=init_w, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    if best_result is None or result.fun < best_result.fun:
        best_result = result

optimal_w = np.abs(best_result.x)
optimal_w = optimal_w / optimal_w.sum()

ensemble_oof  = optimal_w[0] * xgb_oof + optimal_w[1] * lgbm_oof + optimal_w[2] * cat_oof
ensemble_test = optimal_w[0] * xgb_test + optimal_w[1] * lgbm_test + optimal_w[2] * cat_test
ensemble_auc  = roc_auc_score(y, ensemble_oof)

print(f"\n   Optimal Ağırlıklar:")
print(f"     XGBoost  : {optimal_w[0]:.4f}")
print(f"     LightGBM : {optimal_w[1]:.4f}")
print(f"     CatBoost : {optimal_w[2]:.4f}")

# Eşit ağırlıklı karşılaştırma
equal_oof = (xgb_oof + lgbm_oof + cat_oof) / 3.0
equal_auc = roc_auc_score(y, equal_oof)

print(f"\n{'='*50}")
print(f"   🎯 XGBoost  OOF AUC : {xgb_auc:.5f}")
print(f"   🎯 LightGBM OOF AUC : {lgbm_auc:.5f}")
print(f"   🎯 CatBoost OOF AUC : {cat_auc:.5f}")
print(f"   {'─'*40}")
print(f"   📊 Eşit Ağırlıklı   : {equal_auc:.5f}")
print(f"   🏆 Optimal Ensemble  : {ensemble_auc:.5f}")
print(f"{'='*50}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  7. TESTLİM DOSYASI (SUBMISSION)                            ║
# ╚══════════════════════════════════════════════════════════════╝
submission = pd.DataFrame({
    'id': test_ids,
    'Churn': ensemble_test
})
output_file = f'{OUTPUT_DIR}/submission_v4_ultimate.csv'
submission.to_csv(output_file, index=False)

elapsed = time.time() - t0
print(f"\n✅ Teslim dosyası kaydedildi: {output_file}")
print(f"⏱️  Toplam süre: {elapsed/60:.1f} dakika")
print(f"\n🎉 Pipeline tamamlandı! Kaggle'a yükleyin ve skorunuzu görün! 🎉")
