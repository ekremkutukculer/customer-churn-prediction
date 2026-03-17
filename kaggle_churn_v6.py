"""
🏆 CUSTOMER CHURN PREDİCTION — 1st PLACE CHAMPION STRATEGY v6
================================================================
Playground Series S6E3

1. sıra çözümünden ilham alan strateji:
  - 7 farklı feature temsili (binning, digit, frequency, all-cats...)
  - 4 model tipi × 7 feature seti = ~28+ OOF havuzu
  - Optuna ile en iyi OOF alt kümesi seçimi
  - Ridge Regression meta-model (basit & güçlü)
  - Full-data retraining (20 seed ortalaması)
  - Pseudo-labeling YOK (1. sıra: "işe yaramadı")

Kaggle Kullanımı:
  - GPU accelerator açın
  - Yarışma datasını ekleyin
  - Bu scripti çalıştırın
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeClassifierCV, Ridge
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
import time
import os
import gc

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║  YAPILANDIRMA (CONFIGURATION)                               ║
# ╚══════════════════════════════════════════════════════════════╝
N_FOLDS              = 5        # Kazanan 5-fold kullandı
CV_SEED              = 42       # Sabit split — OOF ve stacking aynı fold'lar
RETRAIN_SEEDS        = 20       # Full-data retraining seed sayısı
OPTUNA_SUBSET_TRIALS = 1000     # OOF subset arama denemesi
USE_GPU              = True
USE_OPTUNA_SUBSET    = True     # False → tüm OOF'ları kullan
USE_ORIGINAL_DATA    = True     # Orijinal Telco veri setini kullan

# Kaggle mı yoksa lokal mi?
if os.path.exists('/kaggle/input/playground-series-s6e3/train.csv'):
    DATA_DIR = '/kaggle/input/playground-series-s6e3'
    OUTPUT_DIR = '/kaggle/working'
    # Kaggle'da orijinal veri seti yolu (dataset olarak eklenmiş olmalı)
    ORIG_DATA_PATH = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
else:
    DATA_DIR = 'data'
    OUTPUT_DIR = '.'
    ORIG_DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

DEVICE = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')

print("🏆 CHAMPION STRATEGY PIPELINE v6 BAŞLIYOR 🏆")
print("=" * 75)
print(f"📁 Veri dizini          : {DATA_DIR}")
print(f"📊 Fold sayısı          : {N_FOLDS}")
print(f"🌱 Retraining seeds     : {RETRAIN_SEEDS}")
print(f"🔍 Optuna subset trials : {OPTUNA_SUBSET_TRIALS}")
print(f"🖥️  GPU                  : {DEVICE}")
print(f"📦 Orijinal veri        : {'Açık' if USE_ORIGINAL_DATA else 'Kapalı'}")
print("=" * 75)

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. VERİ YÜKLEME (DATA LOADING)                             ║
# ╚══════════════════════════════════════════════════════════════╝
t0 = time.time()
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')
test_ids = test_df['id'].copy()

train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})

print(f"\n📥 Veri yüklendi: Train={train_df.shape}, Test={test_df.shape}  ({time.time()-t0:.1f}s)")

# ╔══════════════════════════════════════════════════════════════╗
# ║  1.5 ORİJİNAL VERİ SETİ (ORIGINAL DATASET MERGE)            ║
# ║  Kazanan: "extracted statistics from the original data"      ║
# ╚══════════════════════════════════════════════════════════════╝
orig_df = None
if USE_ORIGINAL_DATA and os.path.exists(ORIG_DATA_PATH):
    print("\n📦 Orijinal Telco veri seti yükleniyor...")
    orig_df = pd.read_csv(ORIG_DATA_PATH)

    # Temizlik
    if 'customerID' in orig_df.columns:
        orig_df.drop('customerID', axis=1, inplace=True)
    if 'Churn' in orig_df.columns and orig_df['Churn'].dtype == 'object':
        orig_df['Churn'] = orig_df['Churn'].map({'No': 0, 'Yes': 1})
    if 'TotalCharges' in orig_df.columns and orig_df['TotalCharges'].dtype == 'object':
        orig_df['TotalCharges'] = pd.to_numeric(orig_df['TotalCharges'], errors='coerce')
        orig_df['TotalCharges'].fillna(orig_df['TotalCharges'].median(), inplace=True)

    # Ortak sütunları bul ve birleştir
    common_cols = [col for col in train_df.columns if col in orig_df.columns]
    orig_aligned = orig_df[common_cols].copy()

    n_before = len(train_df)
    train_df = pd.concat([train_df, orig_aligned], ignore_index=True)
    train_df['Churn'] = train_df['Churn'].astype(int)
    train_df['TotalCharges'].fillna(train_df['TotalCharges'].median(), inplace=True)

    print(f"   ✅ Orijinal veri birleştirildi: {n_before:,} → {len(train_df):,} (+{len(orig_aligned):,})")
elif USE_ORIGINAL_DATA:
    print(f"\n⚠️  Orijinal veri bulunamadı: {ORIG_DATA_PATH}")
    print(f"   Kaggle'da 'Add Data' → 'telco-customer-churn' dataset ekleyin.")

y = train_df['Churn'].values
n_train = len(y)
n_test  = len(test_df)

# Sabit KFold — tüm pipeline boyunca aynı split'ler
SKF = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
FOLD_INDICES = list(SKF.split(np.zeros(n_train), y))

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. ÖZELLİK TEMSİLLERİ (FEATURE REPRESENTATIONS)           ║
# ║  Kazananın Bölüm 2: 7 farklı temsil                         ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n🔧 Feature Representations oluşturuluyor...")

drop_cols = ['id', 'Churn']
X_raw = train_df.drop(columns=drop_cols).copy()
X_test_raw = test_df.drop(columns=['id']).copy()

cat_cols_orig = X_raw.select_dtypes(include='object').columns.tolist()
num_cols_orig = X_raw.select_dtypes(exclude='object').columns.tolist()

# ---- Yardımcı fonksiyonlar ----
def label_encode_df(X_tr, X_te, cat_cols):
    """Label encode categorical columns."""
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_tr[col], X_te[col]], axis=0).astype(str)
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
    return X_tr, X_te


def to_float32(X_tr, X_te):
    """Convert to float32 numpy arrays."""
    return X_tr.astype(np.float32).values, X_te.astype(np.float32).values


# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 1: BASE (Label Encoded)                      ║
# ╚══════════════════════════════════════════════════════════════╝
X_base_tr, X_base_te = label_encode_df(X_raw, X_test_raw, cat_cols_orig)
X_base_tr, X_base_te = to_float32(X_base_tr, X_base_te)
print(f"   ✅ REP1 BASE: {X_base_tr.shape[1]} features")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 2: BASE + ENGINEERED FEATURES                ║
# ╚══════════════════════════════════════════════════════════════╝
hizmetler = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
             'TechSupport', 'StreamingTV', 'StreamingMovies']

def add_engineered(df_orig):
    df = df_orig.copy()
    df['Total_Services']       = (df[hizmetler] == 'Yes').sum(axis=1) if hizmetler[0] in df.columns and df[hizmetler[0]].dtype == 'object' else 0
    df['Avg_Monthly_Spend']    = df['TotalCharges'] / (df['tenure'] + 1)
    df['Charge_Per_Service']   = df['MonthlyCharges'] / (df.get('Total_Services', 0) + 1)
    df['Tenure_x_Monthly']     = df['tenure'] * df['MonthlyCharges']
    df['Contract_Value']       = df['tenure'] * df['MonthlyCharges'] - df['TotalCharges']
    df['Monthly_to_Total']     = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['Is_New_Customer']      = (df['tenure'] <= 6).astype(int)
    df['Is_VIP_Customer']      = (df['tenure'] >= 48).astype(int)
    df['Tenure_Squared']       = df['tenure'] ** 2
    df['Tenure_Log']           = np.log1p(df['tenure'])
    df['Monthly_Charge_Diff']  = df['MonthlyCharges'] - df['Avg_Monthly_Spend']
    return df

X_eng_tr_df = add_engineered(X_raw)
X_eng_te_df = add_engineered(X_test_raw)

# Hizmet bazlı feature'lar (object sütunlardan)
for col in hizmetler:
    if col in X_raw.columns:
        X_eng_tr_df[f'{col}_Yes'] = (X_raw[col] == 'Yes').astype(int)
        X_eng_te_df[f'{col}_Yes'] = (X_test_raw[col] == 'Yes').astype(int)

# Interaction features
if 'InternetService' in X_raw.columns:
    X_eng_tr_df['Fiber_Optic'] = (X_raw['InternetService'] == 'Fiber optic').astype(int)
    X_eng_te_df['Fiber_Optic'] = (X_test_raw['InternetService'] == 'Fiber optic').astype(int)
    X_eng_tr_df['No_Internet'] = (X_raw['InternetService'] == 'No').astype(int)
    X_eng_te_df['No_Internet'] = (X_test_raw['InternetService'] == 'No').astype(int)

if 'Contract' in X_raw.columns:
    X_eng_tr_df['Month_to_Month'] = (X_raw['Contract'] == 'Month-to-month').astype(int)
    X_eng_te_df['Month_to_Month'] = (X_test_raw['Contract'] == 'Month-to-month').astype(int)

if 'PaymentMethod' in X_raw.columns:
    X_eng_tr_df['Electronic_Check'] = (X_raw['PaymentMethod'] == 'Electronic check').astype(int)
    X_eng_te_df['Electronic_Check'] = (X_test_raw['PaymentMethod'] == 'Electronic check').astype(int)
    X_eng_tr_df['Auto_Payment'] = X_raw['PaymentMethod'].str.contains('automatic', case=False).astype(int)
    X_eng_te_df['Auto_Payment'] = X_test_raw['PaymentMethod'].str.contains('automatic', case=False).astype(int)

# Encode categoricals
eng_cat_cols = X_eng_tr_df.select_dtypes(include='object').columns.tolist()
X_eng_tr_df, X_eng_te_df = label_encode_df(X_eng_tr_df, X_eng_te_df, eng_cat_cols)
X_eng_tr, X_eng_te = to_float32(X_eng_tr_df, X_eng_te_df)
print(f"   ✅ REP2 ENGINEERED: {X_eng_tr.shape[1]} features")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 3: BASE + BINNING                            ║
# ║  Kazanan: "qcut, cut, rounding"                              ║
# ╚══════════════════════════════════════════════════════════════╝
X_bin_tr_df = X_raw.copy()
X_bin_te_df = X_test_raw.copy()

for col in num_cols_orig:
    train_vals = X_bin_tr_df[col]
    test_vals  = X_bin_te_df[col]

    # Quantile binning (10 bins)
    try:
        bins_q = pd.qcut(train_vals, q=10, duplicates='drop', retbins=True)[1]
        X_bin_tr_df[f'{col}_qbin'] = pd.cut(train_vals, bins=bins_q, labels=False, include_lowest=True).fillna(0).astype(int)
        X_bin_te_df[f'{col}_qbin'] = pd.cut(test_vals, bins=bins_q, labels=False, include_lowest=True).fillna(0).astype(int)
    except:
        pass

    # Equal-width binning (10 bins)
    try:
        bins_e = pd.cut(train_vals, bins=10, retbins=True, include_lowest=True)[1]
        X_bin_tr_df[f'{col}_ebin'] = pd.cut(train_vals, bins=bins_e, labels=False, include_lowest=True).fillna(0).astype(int)
        X_bin_te_df[f'{col}_ebin'] = pd.cut(test_vals, bins=bins_e, labels=False, include_lowest=True).fillna(0).astype(int)
    except:
        pass

    # Rounding
    if train_vals.max() > 10:
        divisor = 5 if train_vals.max() < 100 else 10
        X_bin_tr_df[f'{col}_round'] = (train_vals / divisor).round().astype(int)
        X_bin_te_df[f'{col}_round'] = (test_vals / divisor).round().astype(int)

bin_cat_cols = X_bin_tr_df.select_dtypes(include='object').columns.tolist()
X_bin_tr_df, X_bin_te_df = label_encode_df(X_bin_tr_df, X_bin_te_df, bin_cat_cols)
X_bin_tr, X_bin_te = to_float32(X_bin_tr_df, X_bin_te_df)
print(f"   ✅ REP3 BINNING: {X_bin_tr.shape[1]} features")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 4: BASE + DIGIT FEATURES                     ║
# ║  Kazanan: "units, tens, hundreds, decimal digits"             ║
# ╚══════════════════════════════════════════════════════════════╝
X_dig_tr_df = X_raw.copy()
X_dig_te_df = X_test_raw.copy()

for col in num_cols_orig:
    tr_abs = X_dig_tr_df[col].abs()
    te_abs = X_dig_te_df[col].abs()

    # Integer part digits
    tr_int = tr_abs.astype(int)
    te_int = te_abs.astype(int)
    X_dig_tr_df[f'{col}_units'] = tr_int % 10
    X_dig_te_df[f'{col}_units'] = te_int % 10
    X_dig_tr_df[f'{col}_tens']  = (tr_int // 10) % 10
    X_dig_te_df[f'{col}_tens']  = (te_int // 10) % 10
    if tr_int.max() >= 100:
        X_dig_tr_df[f'{col}_hundreds'] = (tr_int // 100) % 10
        X_dig_te_df[f'{col}_hundreds'] = (te_int // 100) % 10

    # First decimal digit
    decimal_part_tr = ((tr_abs - tr_int) * 10).astype(int)
    decimal_part_te = ((te_abs - te_int) * 10).astype(int)
    X_dig_tr_df[f'{col}_dec1'] = decimal_part_tr
    X_dig_te_df[f'{col}_dec1'] = decimal_part_te

dig_cat_cols = X_dig_tr_df.select_dtypes(include='object').columns.tolist()
X_dig_tr_df, X_dig_te_df = label_encode_df(X_dig_tr_df, X_dig_te_df, dig_cat_cols)
X_dig_tr, X_dig_te = to_float32(X_dig_tr_df, X_dig_te_df)
print(f"   ✅ REP4 DIGIT: {X_dig_tr.shape[1]} features")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 5: ALL-CATS (tüm features → string)          ║
# ║  Kazanan: "all features as categorical"                      ║
# ╚══════════════════════════════════════════════════════════════╝
X_ac_tr_df = X_raw.copy()
X_ac_te_df = X_test_raw.copy()

for col in num_cols_orig:
    X_ac_tr_df[col] = X_ac_tr_df[col].astype(str)
    X_ac_te_df[col] = X_ac_te_df[col].astype(str)

all_ac_cols = X_ac_tr_df.columns.tolist()
X_ac_tr_df, X_ac_te_df = label_encode_df(X_ac_tr_df, X_ac_te_df, all_ac_cols)
X_ac_tr, X_ac_te = to_float32(X_ac_tr_df, X_ac_te_df)
print(f"   ✅ REP5 ALL-CATS: {X_ac_tr.shape[1]} features")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 6: FREQUENCY ENCODING                        ║
# ║  Kazanan: "frequency of each value"                          ║
# ╚══════════════════════════════════════════════════════════════╝
X_freq_tr_df = X_raw.copy()
X_freq_te_df = X_test_raw.copy()

for col in X_freq_tr_df.columns:
    freq_map = X_freq_tr_df[col].value_counts(normalize=True).to_dict()
    X_freq_tr_df[f'{col}_freq'] = X_freq_tr_df[col].map(freq_map).astype(np.float32)
    X_freq_te_df[f'{col}_freq'] = X_freq_te_df[col].map(freq_map).fillna(0).astype(np.float32)

freq_cat_cols = X_freq_tr_df.select_dtypes(include='object').columns.tolist()
X_freq_tr_df, X_freq_te_df = label_encode_df(X_freq_tr_df, X_freq_te_df, freq_cat_cols)
X_freq_tr, X_freq_te = to_float32(X_freq_tr_df, X_freq_te_df)
print(f"   ✅ REP6 FREQUENCY: {X_freq_tr.shape[1]} features")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 7: ORIG_STATS (Orijinal veri istatistikleri)  ║
# ║  Kazanan: "target mean, smoothed mean, WoE, entropy"         ║
# ╚══════════════════════════════════════════════════════════════╝
if orig_df is not None:
    print("   🔬 Orijinal veriden target istatistikleri çıkarılıyor...")
    X_os_tr_df = X_raw.copy()
    X_os_te_df = X_test_raw.copy()

    # Orijinal veri üzerinden istatistik hesapla
    orig_features = orig_df.drop(columns=['Churn'], errors='ignore')
    orig_target   = orig_df['Churn']
    overall_mean  = orig_target.mean()
    overall_count = len(orig_target)

    orig_cat_cols = orig_features.select_dtypes(include='object').columns.tolist()
    orig_num_cols = orig_features.select_dtypes(exclude='object').columns.tolist()

    # Kategorik sütunlar için target statistics
    for col in orig_cat_cols:
        if col not in X_os_tr_df.columns:
            continue
        group = orig_df.groupby(col)['Churn']

        # 1. Target Mean
        target_mean = group.mean().to_dict()
        X_os_tr_df[f'{col}_tmean'] = X_os_tr_df[col].map(target_mean).fillna(overall_mean).astype(np.float32)
        X_os_te_df[f'{col}_tmean'] = X_os_te_df[col].map(target_mean).fillna(overall_mean).astype(np.float32)

        # 2. Smoothed Target Mean (Bayesian smoothing, m=10)
        m = 10
        counts = group.count().to_dict()
        sums   = group.sum().to_dict()
        smooth_map = {k: (sums.get(k, 0) + m * overall_mean) / (counts.get(k, 0) + m)
                      for k in set(list(counts.keys()))}
        X_os_tr_df[f'{col}_smean'] = X_os_tr_df[col].map(smooth_map).fillna(overall_mean).astype(np.float32)
        X_os_te_df[f'{col}_smean'] = X_os_te_df[col].map(smooth_map).fillna(overall_mean).astype(np.float32)

        # 3. Weight of Evidence (WoE)
        total_pos = orig_target.sum()
        total_neg = overall_count - total_pos
        woe_map = {}
        for val, sub in orig_df.groupby(col)['Churn']:
            pos = sub.sum() + 0.5  # Laplace smoothing
            neg = len(sub) - sub.sum() + 0.5
            woe_map[val] = np.log((pos / total_pos) / (neg / total_neg))
        X_os_tr_df[f'{col}_woe'] = X_os_tr_df[col].map(woe_map).fillna(0).astype(np.float32)
        X_os_te_df[f'{col}_woe'] = X_os_te_df[col].map(woe_map).fillna(0).astype(np.float32)

    # Sayısal sütunlar için binned target statistics
    for col in orig_num_cols:
        if col not in X_os_tr_df.columns:
            continue
        try:
            orig_df[f'{col}_qbin_tmp'] = pd.qcut(orig_df[col], q=10, duplicates='drop')
            group = orig_df.groupby(f'{col}_qbin_tmp')['Churn']
            tmean_map = group.mean()

            train_bins = pd.cut(X_os_tr_df[col], bins=pd.qcut(orig_df[col], q=10, duplicates='drop', retbins=True)[1],
                               include_lowest=True)
            test_bins  = pd.cut(X_os_te_df[col], bins=pd.qcut(orig_df[col], q=10, duplicates='drop', retbins=True)[1],
                               include_lowest=True)
            X_os_tr_df[f'{col}_orig_tmean'] = train_bins.map(tmean_map).fillna(overall_mean).astype(np.float32)
            X_os_te_df[f'{col}_orig_tmean'] = test_bins.map(tmean_map).fillna(overall_mean).astype(np.float32)
            orig_df.drop(columns=[f'{col}_qbin_tmp'], inplace=True)
        except:
            pass

    # Encode remaining categoricals
    os_cat_cols = X_os_tr_df.select_dtypes(include='object').columns.tolist()
    X_os_tr_df, X_os_te_df = label_encode_df(X_os_tr_df, X_os_te_df, os_cat_cols)
    X_os_tr, X_os_te = to_float32(X_os_tr_df, X_os_te_df)
    print(f"   ✅ REP7 ORIG_STATS: {X_os_tr.shape[1]} features")
else:
    X_os_tr, X_os_te = None, None
    print("   ⏭️  REP7 ORIG_STATS: Atlandı (orijinal veri yok)")

# ╔══════════════════════════════════════════════════════════════╗
# ║  REPRESENTATION 8: FULL (tüm temsillerin birleşimi)          ║
# ╚══════════════════════════════════════════════════════════════╝
base_offset = len(num_cols_orig) + len(cat_cols_orig)
full_parts = [X_base_tr, X_eng_tr[:, base_offset:],
              X_bin_tr[:, base_offset:], X_dig_tr[:, base_offset:],
              X_freq_tr[:, base_offset:]]
full_parts_te = [X_base_te, X_eng_te[:, base_offset:],
                 X_bin_te[:, base_offset:], X_dig_te[:, base_offset:],
                 X_freq_te[:, base_offset:]]
if X_os_tr is not None:
    full_parts.append(X_os_tr[:, base_offset:])
    full_parts_te.append(X_os_te[:, base_offset:])

X_full_tr = np.hstack(full_parts)
X_full_te = np.hstack(full_parts_te)
print(f"   ✅ REP8 FULL: {X_full_tr.shape[1]} features")

# Feature representations dictionary
REPRESENTATIONS = {
    'BASE':       (X_base_tr, X_base_te),
    'ENGINEERED': (X_eng_tr, X_eng_te),
    'BINNING':    (X_bin_tr, X_bin_te),
    'DIGIT':      (X_dig_tr, X_dig_te),
    'ALL_CATS':   (X_ac_tr, X_ac_te),
    'FREQUENCY':  (X_freq_tr, X_freq_te),
    'FULL':       (X_full_tr, X_full_te),
}
if X_os_tr is not None:
    REPRESENTATIONS['ORIG_STATS'] = (X_os_tr, X_os_te)

print(f"\n   📊 Toplam {len(REPRESENTATIONS)} feature temsili hazır.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  3. OOF HAVUZzU OLUŞTURMA (OOF POOL GENERATION)             ║
# ║  Her model × her feature seti = çok sayıda OOF               ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n{'='*75}")
print("🏊 OOF HAVUZU OLUŞTURULUYOR 🏊")
print(f"{'='*75}")

# OOF ve test tahminlerini saklayan sözlük
oof_pool = {}       # name → np.array(n_train,)
test_pool = {}      # name → np.array(n_test,)
best_iters = {}     # name → avg best iteration (for full-data retraining)

def generate_oof(name, X_tr, y, X_te, create_model_fn, model_type='tree'):
    """Sabit fold split'lerle OOF ve test tahminleri üretir."""
    oof_pred  = np.zeros(n_train)
    test_pred = np.zeros(n_test)
    iterations = []

    for fold, (tr_idx, val_idx) in enumerate(FOLD_INDICES):

        if model_type == 'tree':
            model = create_model_fn()
            if 'XGB' in name:
                model.fit(X_tr[tr_idx], y[tr_idx],
                          eval_set=[(X_tr[val_idx], y[val_idx])], verbose=False)
                iterations.append(model.best_iteration if hasattr(model, 'best_iteration') else
                                  model.get_booster().best_iteration)
            elif 'LGBM' in name:
                model.fit(X_tr[tr_idx], y[tr_idx],
                          eval_set=[(X_tr[val_idx], y[val_idx])])
                iterations.append(model.best_iteration_)
            elif 'CAT' in name:
                model.fit(X_tr[tr_idx], y[tr_idx],
                          eval_set=[(X_tr[val_idx], y[val_idx])], verbose=False)
                iterations.append(model.get_best_iteration())

            oof_pred[val_idx] = model.predict_proba(X_tr[val_idx])[:, 1]
            test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS

        elif model_type == 'mlp':
            # MLP: fold bazında eğitim
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr[tr_idx])
            X_val_s = scaler.transform(X_tr[val_idx])
            X_te_s  = scaler.transform(X_te)

            fold_oof, fold_test = _train_mlp(X_tr_s, y[tr_idx], X_val_s, y[val_idx],
                                              X_te_s, seed=CV_SEED + fold)
            oof_pred[val_idx] = fold_oof
            test_pred += fold_test / N_FOLDS

    auc = roc_auc_score(y, oof_pred)
    oof_pool[name] = oof_pred
    test_pool[name] = test_pred
    if iterations:
        best_iters[name] = int(np.mean(iterations))
    print(f"   🎯 {name:<35s} OOF AUC: {auc:.5f}")
    return auc


# ---- MLP Helper ----
class ChurnMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


def _train_mlp(X_tr, y_tr, X_val, y_val, X_te, seed=42, epochs=50, lr=1e-3, bs=1024):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    X_te_t  = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    model = ChurnMLP(X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.BCEWithLogitsLoss()

    best_auc, best_state, patience = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        vauc = roc_auc_score(y_val, vp)
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oof  = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        test = torch.sigmoid(model(X_te_t)).cpu().numpy().flatten()
    return oof, test


# ╔══════════════════════════════════════════════════════════════╗
# ║  OOF ÜRETİMİ: MODEL × REPRESENTATION                        ║
# ╚══════════════════════════════════════════════════════════════╝

# ---- XGBoost configs ----
xgb_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 6,
     'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5, 'gamma': 0.1,
     'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'n_estimators': 2000, 'learning_rate': 0.05, 'max_depth': 4,
     'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 10, 'gamma': 0.5,
     'reg_alpha': 1.0, 'reg_lambda': 5.0},
]

# ---- LightGBM configs ----
lgbm_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 7,
     'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_samples': 20, 'num_leaves': 63,
     'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'n_estimators': 2000, 'learning_rate': 0.05, 'max_depth': 5,
     'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_samples': 50, 'num_leaves': 31,
     'reg_alpha': 1.0, 'reg_lambda': 5.0},
]

# ---- CatBoost configs ----
cat_configs = [
    {'iterations': 2000, 'learning_rate': 0.03, 'depth': 7,
     'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5, 'random_strength': 0.5},
    {'iterations': 2000, 'learning_rate': 0.05, 'depth': 5,
     'l2_leaf_reg': 5.0, 'bagging_temperature': 1.0, 'random_strength': 1.0},
]

print(f"\n--- XGBoost OOF'ları ---")
for rep_name, (X_tr, X_te) in REPRESENTATIONS.items():
    for cfg_idx, cfg in enumerate(xgb_configs):
        name = f"XGB_c{cfg_idx}_{rep_name}"
        def make_xgb(c=cfg):
            kw = {**c, 'tree_method': 'hist', 'random_state': CV_SEED,
                  'eval_metric': 'auc', 'early_stopping_rounds': 100, 'verbosity': 0}
            if USE_GPU:
                kw['device'] = 'cuda'
            return XGBClassifier(**kw)
        generate_oof(name, X_tr, y, X_te, make_xgb, 'tree')

print(f"\n--- LightGBM OOF'ları ---")
for rep_name, (X_tr, X_te) in REPRESENTATIONS.items():
    for cfg_idx, cfg in enumerate(lgbm_configs):
        name = f"LGBM_c{cfg_idx}_{rep_name}"
        def make_lgbm(c=cfg):
            return LGBMClassifier(**c, random_state=CV_SEED, verbose=-1,
                                   early_stopping_rounds=100)
        generate_oof(name, X_tr, y, X_te, make_lgbm, 'tree')

print(f"\n--- CatBoost OOF'ları ---")
for rep_name, (X_tr, X_te) in REPRESENTATIONS.items():
    for cfg_idx, cfg in enumerate(cat_configs):
        name = f"CAT_c{cfg_idx}_{rep_name}"
        def make_cat(c=cfg):
            kw = {**c, 'random_seed': CV_SEED, 'verbose': False,
                  'eval_metric': 'AUC', 'early_stopping_rounds': 100}
            if USE_GPU:
                kw['task_type'] = 'GPU'
            return CatBoostClassifier(**kw)
        generate_oof(name, X_tr, y, X_te, make_cat, 'tree')

print(f"\n--- MLP OOF'ları ---")
# MLP sadece sayısal-ağırlıklı setlerde iyi çalışır
mlp_reps = ['BASE', 'ENGINEERED', 'FULL', 'FREQUENCY']
if 'ORIG_STATS' in REPRESENTATIONS:
    mlp_reps.append('ORIG_STATS')
for rep_name in mlp_reps:
    X_tr, X_te = REPRESENTATIONS[rep_name]
    name = f"MLP_{rep_name}"
    generate_oof(name, X_tr, y, X_te, None, 'mlp')

gc.collect()
total_oofs = len(oof_pool)
print(f"\n📊 Toplam OOF sayısı: {total_oofs}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  4. OPTUNA İLE OOF SUBSET SEÇİMİ                            ║
# ║  Kazanan: "2500 trials, only ~1/10 selected"                 ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n{'='*75}")
print("🔍 OPTUNA OOF SUBSET SEÇİMİ 🔍")
print(f"{'='*75}")

oof_names = list(oof_pool.keys())
oof_matrix = np.column_stack([oof_pool[n] for n in oof_names])  # (n_train, n_oofs)
test_matrix = np.column_stack([test_pool[n] for n in oof_names])

if USE_OPTUNA_SUBSET:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def subset_objective(trial):
        # Her trial rastgele bir alt küme seçer
        selected = []
        for i, name in enumerate(oof_names):
            if trial.suggest_categorical(f'use_{i}', [True, False]):
                selected.append(i)

        if len(selected) < 3:
            return 0.0  # Çok az model seçildi

        # Ridge ile birleştir (KFold-safe)
        X_stack = oof_matrix[:, selected]
        oof_ridge = np.zeros(n_train)

        for tr_idx, val_idx in FOLD_INDICES:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_stack[tr_idx], y[tr_idx])
            oof_ridge[val_idx] = ridge.predict(X_stack[val_idx])

        return roc_auc_score(y, oof_ridge)

    print(f"   ⏳ {OPTUNA_SUBSET_TRIALS} deneme başlıyor...")
    study = optuna.create_study(direction='maximize')
    study.optimize(subset_objective, n_trials=OPTUNA_SUBSET_TRIALS, show_progress_bar=True)

    # En iyi subset
    best_mask = [study.best_params[f'use_{i}'] for i in range(len(oof_names))]
    selected_names = [n for n, m in zip(oof_names, best_mask) if m]
    selected_indices = [i for i, m in enumerate(best_mask) if m]

    print(f"\n   ✅ En iyi AUC: {study.best_value:.5f}")
    print(f"   📊 Seçilen OOF sayısı: {len(selected_names)} / {total_oofs}")
    print(f"   📋 Seçilenler:")
    for name in selected_names:
        print(f"      • {name}")
else:
    selected_names = oof_names
    selected_indices = list(range(len(oof_names)))
    print(f"   ⏭️  Optuna kapalı — tüm {total_oofs} OOF kullanılıyor.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  5. RIDGE REGRESSION STACKING                                ║
# ║  Kazanan: "Ridge worked best — simple, stable, robust"       ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n{'='*75}")
print("🏔️  RIDGE REGRESSION STACKİNG 🏔️")
print(f"{'='*75}")

X_stack = oof_matrix[:, selected_indices]
X_stack_test = test_matrix[:, selected_indices]

# Ridge alpha optimizasyonu — basit grid search
best_ridge_auc = 0.0
best_ridge_alpha = 1.0
best_ridge_oof = None

for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    oof_ridge = np.zeros(n_train)
    for tr_idx, val_idx in FOLD_INDICES:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_stack[tr_idx], y[tr_idx])
        oof_ridge[val_idx] = ridge.predict(X_stack[val_idx])

    auc = roc_auc_score(y, oof_ridge)
    if auc > best_ridge_auc:
        best_ridge_auc = auc
        best_ridge_alpha = alpha
        best_ridge_oof = oof_ridge

print(f"   ✅ En iyi Ridge alpha: {best_ridge_alpha}")
print(f"   🎯 Ridge OOF AUC: {best_ridge_auc:.5f}")

# Final Ridge — tüm train üzerinde eğit
final_ridge = Ridge(alpha=best_ridge_alpha)
final_ridge.fit(X_stack, y)
ridge_test_pred = final_ridge.predict(X_stack_test)

# Clip to [0, 1]
ridge_test_pred = np.clip(ridge_test_pred, 0, 1)

# ╔══════════════════════════════════════════════════════════════╗
# ║  6. FULL-DATA RETRAINING (20 SEED)                           ║
# ║  Kazanan: "retrain on full data, 20 seeds, 1.25x iterations" ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n{'='*75}")
print(f"🔄 FULL-DATA RETRAINING ({RETRAIN_SEEDS} seed) 🔄")
print(f"{'='*75}")

# Seçilen ağaç modellerini full-data ile tekrar eğit
retrained_test_preds = {}

for name in selected_names:
    if name not in best_iters:
        continue  # MLP için full-data retraining yapmıyoruz

    avg_iter = best_iters[name]
    full_n_estimators = int(avg_iter * 1.25)

    # Hangi representation?
    rep_name = name.split('_', 2)[-1]  # "XGB_c0_BASE" → "BASE"
    if rep_name not in REPRESENTATIONS:
        continue
    X_tr_full, X_te_full = REPRESENTATIONS[rep_name]

    seed_preds = []
    for seed in range(RETRAIN_SEEDS):
        if 'XGB' in name:
            cfg_idx = int(name.split('_')[1][1])
            cfg = {**xgb_configs[cfg_idx], 'n_estimators': full_n_estimators}
            kw = {'tree_method': 'hist', 'random_state': seed, 'eval_metric': 'auc', 'verbosity': 0}
            if USE_GPU:
                kw['device'] = 'cuda'
            model = XGBClassifier(**cfg, **kw)
            model.fit(X_tr_full, y, verbose=False)
            seed_preds.append(model.predict_proba(X_te_full)[:, 1])

        elif 'LGBM' in name:
            cfg_idx = int(name.split('_')[1][1])
            cfg = {**lgbm_configs[cfg_idx], 'n_estimators': full_n_estimators}
            model = LGBMClassifier(**cfg, random_state=seed, verbose=-1)
            model.fit(X_tr_full, y)
            seed_preds.append(model.predict_proba(X_te_full)[:, 1])

        elif 'CAT' in name:
            cfg_idx = int(name.split('_')[1][1])
            cfg = {**cat_configs[cfg_idx], 'iterations': full_n_estimators}
            kw = {'random_seed': seed, 'verbose': False, 'eval_metric': 'AUC'}
            if USE_GPU:
                kw['task_type'] = 'GPU'
            model = CatBoostClassifier(**cfg, **kw)
            model.fit(X_tr_full, y, verbose=False)
            seed_preds.append(model.predict_proba(X_te_full)[:, 1])

    if seed_preds:
        retrained_test_preds[name] = np.mean(seed_preds, axis=0)
        print(f"   ✅ {name:<35s} → {full_n_estimators} iters × {len(seed_preds)} seeds")

# Full-data test matrix yeniden oluştur
if retrained_test_preds:
    test_matrix_retrained = test_matrix.copy()
    for i, name in enumerate(oof_names):
        if name in retrained_test_preds:
            test_matrix_retrained[:, i] = retrained_test_preds[name]

    X_stack_test_retrained = test_matrix_retrained[:, selected_indices]

    # Ridge ile final tahmin
    final_ridge_retrained = Ridge(alpha=best_ridge_alpha)
    final_ridge_retrained.fit(X_stack, y)  # OOF'lar aynı
    retrained_test_pred = final_ridge_retrained.predict(X_stack_test_retrained)
    retrained_test_pred = np.clip(retrained_test_pred, 0, 1)
    final_test_pred = retrained_test_pred
    print(f"\n   🏆 Full-data retraining uygulandı.")
else:
    final_test_pred = ridge_test_pred
    print(f"\n   ⚠️  Retrain edilecek model bulunamadı, KFold tahminleri kullanılıyor.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  7. SONUÇ RAPORU VE SUBMISSION                               ║
# ╚══════════════════════════════════════════════════════════════╝
# Basit ortalama ile karşılaştırma
simple_avg_oof = oof_matrix[:, selected_indices].mean(axis=1)
simple_avg_auc = roc_auc_score(y, simple_avg_oof)

print(f"\n{'='*75}")
print("📊 FİNAL SONUÇ RAPORU")
print(f"{'='*75}")
print(f"\n   Toplam üretilen OOF  : {total_oofs}")
print(f"   Seçilen OOF          : {len(selected_names)}")
print(f"   {'─'*50}")
print(f"   Basit Ortalama AUC   : {simple_avg_auc:.5f}")
print(f"   🏔️  Ridge Stacking AUC : {best_ridge_auc:.5f}")
print(f"   Ridge alpha          : {best_ridge_alpha}")
if retrained_test_preds:
    print(f"   Full-data retraining : ✅ ({RETRAIN_SEEDS} seed)")
print(f"{'='*75}")

# Top 10 OOF'ları göster
print(f"\n   🏅 En İyi 10 Bireysel OOF:")
oof_aucs = {n: roc_auc_score(y, oof_pool[n]) for n in oof_names}
for i, (n, a) in enumerate(sorted(oof_aucs.items(), key=lambda x: -x[1])[:10], 1):
    marker = "⭐" if n in selected_names else "  "
    print(f"   {marker} {i:>2d}. {n:<35s} AUC: {a:.5f}")

# Submission
submission = pd.DataFrame({
    'id': test_ids,
    'Churn': final_test_pred
})
output_file = f'{OUTPUT_DIR}/submission_v6_champion.csv'
submission.to_csv(output_file, index=False)

elapsed = time.time() - t0
print(f"\n✅ Teslim dosyası: {output_file}")
print(f"⏱️  Toplam süre: {elapsed/60:.1f} dakika")
print(f"\n🏆 Champion Strategy Pipeline tamamlandı! 🏆")
