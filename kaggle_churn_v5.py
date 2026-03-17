"""
🏴‍☠️ CUSTOMER CHURN PREDİCTION — DARK ARTS PIPELINE v5
============================================================
Playground Series S6E3

4 Karanlık Sanat (Dark Art):
  1. Nelder-Mead AUC Weight Optimization (+ Stacking meta-model yarışması)
  2. PyTorch MLP Neural Network (Ağaçlardan farklı bir "beyin")
  3. Feature Diversity (Her modele farklı veri seti)
  4. Pseudo-Labeling (Test verisini "çalma")

Altyapı (v4'ten devralınan):
  - Kapsamlı Feature Engineering (30+ özellik)
  - StratifiedKFold (10-Fold)
  - Multi-Seed Averaging (5 seed)
  - Optuna Hyperparameter Tuning

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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import warnings
import time
import os

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║  YAPILANDIRMA (CONFIGURATION)                               ║
# ╚══════════════════════════════════════════════════════════════╝
N_FOLDS             = 10
SEEDS               = [42, 123, 456, 789, 2024]
OPTUNA_TRIALS       = 60       # HP tuning deneme sayısı
USE_OPTUNA          = True     # False → varsayılan parametreler
USE_GPU             = True     # Kaggle GPU
PSEUDO_LABEL_ROUNDS = 1        # 0 = Pseudo-labeling kapalı
PL_HIGH_THRESHOLD   = 0.95     # Churn=1 eşiği
PL_LOW_THRESHOLD    = 0.05     # Churn=0 eşiği
MLP_EPOCHS          = 50
MLP_LR              = 1e-3
MLP_BATCH_SIZE      = 1024

# Kaggle mı yoksa lokal mi?
if os.path.exists('/kaggle/input/playground-series-s6e3/train.csv'):
    DATA_DIR = '/kaggle/input/playground-series-s6e3'
    OUTPUT_DIR = '/kaggle/working'
else:
    DATA_DIR = 'data'
    OUTPUT_DIR = '.'

DEVICE = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')

print("🏴‍☠️ DARK ARTS PIPELINE v5 BAŞLIYOR 🏴‍☠️")
print("=" * 75)
print(f"📁 Veri dizini       : {DATA_DIR}")
print(f"📊 Fold sayısı       : {N_FOLDS}")
print(f"🌱 Seed sayısı       : {len(SEEDS)}")
print(f"🎯 Optuna            : {'Açık' if USE_OPTUNA else 'Kapalı'} ({OPTUNA_TRIALS} deneme)")
print(f"🖥️  GPU               : {'Açık' if USE_GPU else 'Kapalı'} ({DEVICE})")
print(f"👻 Pseudo-Labeling   : {'Açık' if PSEUDO_LABEL_ROUNDS > 0 else 'Kapalı'} ({PSEUDO_LABEL_ROUNDS} round)")
print(f"🧠 MLP               : {MLP_EPOCHS} epoch, lr={MLP_LR}")
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
X_full = train_df.drop(columns=drop_cols)
X_test_full = test_df.drop(columns=['id'])

cat_cols = X_full.select_dtypes(include='object').columns.tolist()
num_cols = X_full.select_dtypes(exclude='object').columns.tolist()

# Label Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_full[col], X_test_full[col]], axis=0).astype(str)
    le.fit(combined)
    X_full[col] = le.transform(X_full[col].astype(str))
    X_test_full[col] = le.transform(X_test_full[col].astype(str))
    label_encoders[col] = le

# Target Encoding (KFold-safe)
print("   🎯 Target Encoding (KFold-safe)...")
te_cols_source = ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines']
te_feature_names = []
for col in te_cols_source:
    te_col_name = f'{col}_TE'
    te_feature_names.append(te_col_name)
    X_full[te_col_name] = np.nan
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf_te.split(X_full, y):
        means = pd.Series(y[tr_idx]).groupby(X_full[col].iloc[tr_idx].values).mean()
        X_full.iloc[val_idx, X_full.columns.get_loc(te_col_name)] = X_full[col].iloc[val_idx].map(means)
    global_means = pd.Series(y).groupby(X_full[col].values).mean()
    X_test_full[te_col_name] = X_test_full[col].map(global_means)
    overall_mean = y.mean()
    X_full[te_col_name].fillna(overall_mean, inplace=True)
    X_test_full[te_col_name].fillna(overall_mean, inplace=True)

X_full = X_full.astype(np.float32)
X_test_full = X_test_full.astype(np.float32)

# ╔══════════════════════════════════════════════════════════════╗
# ║  🎭 DARK ART #3: FEATURE DIVERSITY                          ║
# ║  Her modele farklı feature seti verilir                      ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n🎭 Feature Diversity hazırlanıyor...")

all_feature_names = X_full.columns.tolist()

# Mühendislik özellikleri (interaction features)
engineered_features = [
    'Total_Services', 'Total_No_Services', 'Has_Any_Service',
    'Avg_Monthly_Spend', 'Charge_Per_Service', 'Tenure_x_Monthly',
    'Contract_Value', 'Monthly_to_Total', 'Is_Overpaying', 'Monthly_Charge_Diff',
    'Is_New_Customer', 'Is_VIP_Customer', 'Tenure_Squared', 'Tenure_Log', 'Tenure_Group',
    'Has_No_Support', 'Has_Streaming', 'Has_Partner_Dep', 'No_Internet', 'Fiber_Optic',
    'Auto_Payment', 'Electronic_Check', 'Month_to_Month', 'Paperless',
    'New_And_MonthToMonth', 'Fiber_NoSupport', 'Senior_MonthToMonth',
    'Electronic_MonthToMonth', 'Senior_NoSupport', 'New_Fiber',
    'Charges_x_NoSupport', 'New_HighCharge',
]

# Baseline = orijinal sütunlar (mühendislik ve TE hariç)
baseline_features = [f for f in all_feature_names
                     if f not in engineered_features and f not in te_feature_names]

# XGBoost: Sadece baseline
xgb_features = baseline_features
# LightGBM: Baseline + mühendislik özellikleri (TE hariç)
lgbm_features = [f for f in all_feature_names if f not in te_feature_names]
# CatBoost: Baseline + target encoding (mühendislik özellikleri hariç)
cat_features = baseline_features + te_feature_names
# MLP: TÜM özellikler
mlp_features = all_feature_names

print(f"   XGBoost  features: {len(xgb_features)}")
print(f"   LightGBM features: {len(lgbm_features)}")
print(f"   CatBoost  features: {len(cat_features)}")
print(f"   MLP       features: {len(mlp_features)}")

# Feature setlerini numpy'a çevir
X_xgb       = X_full[xgb_features].values
X_xgb_test  = X_test_full[xgb_features].values
X_lgbm      = X_full[lgbm_features].values
X_lgbm_test = X_test_full[lgbm_features].values
X_cat       = X_full[cat_features].values
X_cat_test  = X_test_full[cat_features].values
X_mlp       = X_full[mlp_features].values
X_mlp_test  = X_test_full[mlp_features].values

# ╔══════════════════════════════════════════════════════════════╗
# ║  4. OPTUNA HYPERPARAMETER TUNING                             ║
# ╚══════════════════════════════════════════════════════════════╝
default_xgb_params = {
    'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 5, 'gamma': 0.1,
}
default_lgbm_params = {
    'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 7,
    'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 20, 'num_leaves': 63,
}
default_cat_params = {
    'iterations': 1500, 'learning_rate': 0.03, 'depth': 7,
    'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5, 'random_strength': 0.5,
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
        for tr_idx, val_idx in skf_opt.split(X_xgb, y):
            kw = {'tree_method': 'hist', 'random_state': 42, 'eval_metric': 'auc',
                  'early_stopping_rounds': 100, 'verbosity': 0}
            if USE_GPU:
                kw['device'] = 'cuda'
            model = XGBClassifier(**params, **kw)
            model.fit(X_xgb[tr_idx], y[tr_idx], eval_set=[(X_xgb[val_idx], y[val_idx])], verbose=False)
            scores.append(roc_auc_score(y[val_idx], model.predict_proba(X_xgb[val_idx])[:, 1]))
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
        for tr_idx, val_idx in skf_opt.split(X_lgbm, y):
            model = LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(X_lgbm[tr_idx], y[tr_idx], eval_set=[(X_lgbm[val_idx], y[val_idx])])
            scores.append(roc_auc_score(y[val_idx], model.predict_proba(X_lgbm[val_idx])[:, 1]))
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
        for tr_idx, val_idx in skf_opt.split(X_cat, y):
            kw = {'random_seed': 42, 'verbose': False, 'eval_metric': 'AUC',
                  'early_stopping_rounds': 100}
            if USE_GPU:
                kw['task_type'] = 'GPU'
            model = CatBoostClassifier(**params, **kw)
            model.fit(X_cat[tr_idx], y[tr_idx], eval_set=[(X_cat[val_idx], y[val_idx])], verbose=False)
            scores.append(roc_auc_score(y[val_idx], model.predict_proba(X_cat[val_idx])[:, 1]))
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
# ║  🧠 DARK ART #2: PYTORCH MLP NEURAL NETWORK                 ║
# ╚══════════════════════════════════════════════════════════════╝
class ChurnMLP(nn.Module):
    """Churn tahmini için 3 katmanlı MLP."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_te, seed, epochs=MLP_EPOCHS, lr=MLP_LR, batch_size=MLP_BATCH_SIZE):
    """Tek bir fold için MLP eğitimi. OOF ve test tahminlerini döner."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # StandardScaler — fold bazında fit
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    # Tensörlere çevir
    train_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation tensörleri (GPU'da tut — her epoch tekrar taşıma)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE)
    X_te_t  = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)

    model = ChurnMLP(X_tr_s.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation AUC — gerçek y_val ile
        model.eval()
        with torch.no_grad():
            val_preds = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        val_auc = roc_auc_score(y_val, val_preds)

        # AUC-based early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Best modeli yükle
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oof_pred  = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
        test_pred = torch.sigmoid(model(X_te_t)).cpu().numpy().flatten()

    return oof_pred, test_pred


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. ANA EĞİTİM FONKSİYONU (MAIN TRAINING FUNCTION)         ║
# ╚══════════════════════════════════════════════════════════════╝
def run_full_training(X_xgb, X_lgbm, X_cat, X_mlp, y,
                      X_xgb_test, X_lgbm_test, X_cat_test, X_mlp_test,
                      round_label="ROUND 1"):
    """
    Tüm 4 modeli multi-seed KFold ile eğitir.
    OOF ve test tahminlerini döner.
    """
    print(f"\n{'='*75}")
    print(f"👑 {round_label}: MULTI-SEED KFOLD EĞİTİMİ 👑")
    print(f"{'='*75}")

    # ------ Genel KFold eğitim fonksiyonu (ağaç modelleri için) ------
    def train_tree_kfold(model_name, create_fn, X_tr, y_tr, X_te, n_folds, seeds):
        all_oof  = np.zeros((len(seeds), len(y_tr)))
        all_test = np.zeros((len(seeds), X_te.shape[0]))

        for s_idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            oof_pred  = np.zeros(len(y_tr))
            test_pred = np.zeros(X_te.shape[0])

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
                model = create_fn(seed)
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

    # ------ MLP KFold eğitimi ------
    def train_mlp_kfold(X_tr, y_tr, X_te, n_folds, seeds):
        all_oof  = np.zeros((len(seeds), len(y_tr)))
        all_test = np.zeros((len(seeds), X_te.shape[0]))

        for s_idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            oof_pred  = np.zeros(len(y_tr))
            test_pred = np.zeros(X_te.shape[0])

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
                fold_oof, fold_test = train_mlp_fold(
                    X_tr[tr_idx], y_tr[tr_idx], X_tr[val_idx], y_tr[val_idx], X_te, seed + fold
                )
                oof_pred[val_idx] = fold_oof
                test_pred += fold_test / n_folds

            fold_auc = roc_auc_score(y_tr, oof_pred)
            print(f"   🌱 Seed {seed:>5d} → OOF AUC: {fold_auc:.5f}")
            all_oof[s_idx]  = oof_pred
            all_test[s_idx] = test_pred

        mean_oof  = all_oof.mean(axis=0)
        mean_test = all_test.mean(axis=0)
        mean_auc  = roc_auc_score(y_tr, mean_oof)
        print(f"   📊 MLP Ortalama OOF AUC: {mean_auc:.5f}")
        return mean_oof, mean_test, mean_auc

    # ===== XGBoost =====
    print(f"\n⚡ XGBoost eğitiliyor ({len(SEEDS)}s × {N_FOLDS}f) — {len(xgb_features)} feature...")
    def create_xgb(seed):
        kw = {**best_xgb_params, 'tree_method': 'hist', 'random_state': seed,
              'eval_metric': 'auc', 'early_stopping_rounds': 100, 'verbosity': 0}
        if USE_GPU:
            kw['device'] = 'cuda'
        return XGBClassifier(**kw)
    xgb_oof, xgb_test, xgb_auc = train_tree_kfold('XGBoost', create_xgb, X_xgb, y, X_xgb_test, N_FOLDS, SEEDS)

    # ===== LightGBM =====
    print(f"\n⚡ LightGBM eğitiliyor ({len(SEEDS)}s × {N_FOLDS}f) — {len(lgbm_features)} feature...")
    def create_lgbm(seed):
        return LGBMClassifier(**best_lgbm_params, random_state=seed, verbose=-1,
                               early_stopping_rounds=100)
    lgbm_oof, lgbm_test, lgbm_auc = train_tree_kfold('LightGBM', create_lgbm, X_lgbm, y, X_lgbm_test, N_FOLDS, SEEDS)

    # ===== CatBoost =====
    print(f"\n⚡ CatBoost eğitiliyor ({len(SEEDS)}s × {N_FOLDS}f) — {len(cat_features)} feature...")
    def create_cat(seed):
        kw = {**best_cat_params, 'random_seed': seed, 'verbose': False,
              'eval_metric': 'AUC', 'early_stopping_rounds': 100}
        if USE_GPU:
            kw['task_type'] = 'GPU'
        return CatBoostClassifier(**kw)
    cat_oof, cat_test, cat_auc = train_tree_kfold('CatBoost', create_cat, X_cat, y, X_cat_test, N_FOLDS, SEEDS)

    # ===== MLP (DARK ART #2) =====
    print(f"\n🧠 MLP eğitiliyor ({len(SEEDS)}s × {N_FOLDS}f) — {len(mlp_features)} feature, {MLP_EPOCHS} epoch...")
    mlp_oof, mlp_test, mlp_auc = train_mlp_kfold(X_mlp, y, X_mlp_test, N_FOLDS, SEEDS)

    return (xgb_oof, xgb_test, xgb_auc,
            lgbm_oof, lgbm_test, lgbm_auc,
            cat_oof, cat_test, cat_auc,
            mlp_oof, mlp_test, mlp_auc)


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. ROUND 1 — İLK EĞİTİM                                   ║
# ╚══════════════════════════════════════════════════════════════╝
(xgb_oof, xgb_test, xgb_auc,
 lgbm_oof, lgbm_test, lgbm_auc,
 cat_oof, cat_test, cat_auc,
 mlp_oof, mlp_test, mlp_auc) = run_full_training(
    X_xgb, X_lgbm, X_cat, X_mlp, y,
    X_xgb_test, X_lgbm_test, X_cat_test, X_mlp_test,
    round_label="ROUND 1 (Orijinal Veri)"
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  🔧 DARK ART #1: NELDER-MEAD + STACKING YARIŞMASI           ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n{'='*75}")
print("🏆 DARK ART #1: NELDER-MEAD vs STACKING 🏆")
print(f"{'='*75}")

# --- Yöntem A: Nelder-Mead AUC Optimization ---
def neg_auc_4(weights):
    w = np.abs(weights)
    w = w / w.sum()
    blend = w[0]*xgb_oof + w[1]*lgbm_oof + w[2]*cat_oof + w[3]*mlp_oof
    return -roc_auc_score(y, blend)

best_result = None
for init_w in [[0.25]*4, [0.35, 0.30, 0.25, 0.10], [0.30, 0.25, 0.35, 0.10],
               [0.25, 0.35, 0.30, 0.10], [0.40, 0.20, 0.30, 0.10],
               [0.20, 0.30, 0.20, 0.30], [0.30, 0.30, 0.30, 0.10]]:
    result = minimize(neg_auc_4, x0=init_w, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    if best_result is None or result.fun < best_result.fun:
        best_result = result

nm_w = np.abs(best_result.x)
nm_w = nm_w / nm_w.sum()
nm_oof  = nm_w[0]*xgb_oof + nm_w[1]*lgbm_oof + nm_w[2]*cat_oof + nm_w[3]*mlp_oof
nm_test = nm_w[0]*xgb_test + nm_w[1]*lgbm_test + nm_w[2]*cat_test + nm_w[3]*mlp_test
nm_auc  = roc_auc_score(y, nm_oof)

print(f"\n   🔬 Nelder-Mead Optimal Ağırlıklar:")
print(f"     XGBoost  : {nm_w[0]:.4f}")
print(f"     LightGBM : {nm_w[1]:.4f}")
print(f"     CatBoost : {nm_w[2]:.4f}")
print(f"     MLP      : {nm_w[3]:.4f}")
print(f"   🎯 Nelder-Mead AUC: {nm_auc:.5f}")

# --- Yöntem B: Stacking Meta-Model (LogisticRegression) ---
print("\n   📚 Stacking (Lojistik Regresyon) meta-model eğitiliyor...")
stack_oof_features = np.column_stack([xgb_oof, lgbm_oof, cat_oof, mlp_oof])
stack_test_features = np.column_stack([xgb_test, lgbm_test, cat_test, mlp_test])

# KFold stacking — data leakage önleme
stack_oof_pred = np.zeros(len(y))
stack_test_pred = np.zeros(len(xgb_test))
skf_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in skf_stack.split(stack_oof_features, y):
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(stack_oof_features[tr_idx], y[tr_idx])
    stack_oof_pred[val_idx] = meta_model.predict_proba(stack_oof_features[val_idx])[:, 1]
    stack_test_pred += meta_model.predict_proba(stack_test_features)[:, 1] / 5

stack_auc = roc_auc_score(y, stack_oof_pred)
print(f"   🎯 Stacking AUC: {stack_auc:.5f}")

# --- Kazanan seçimi ---
print(f"\n   {'─'*50}")
if nm_auc >= stack_auc:
    print(f"   🏅 KAZANAN: Nelder-Mead ({nm_auc:.5f} >= {stack_auc:.5f})")
    r1_oof  = nm_oof
    r1_test = nm_test
    r1_auc  = nm_auc
    winner  = "Nelder-Mead"
else:
    print(f"   🏅 KAZANAN: Stacking ({stack_auc:.5f} > {nm_auc:.5f})")
    r1_oof  = stack_oof_pred
    r1_test = stack_test_pred
    r1_auc  = stack_auc
    winner  = "Stacking"

# ╔══════════════════════════════════════════════════════════════╗
# ║  👻 DARK ART #4: PSEUDO-LABELING                            ║
# ╚══════════════════════════════════════════════════════════════╝
final_test_pred = r1_test
final_auc = r1_auc

if PSEUDO_LABEL_ROUNDS > 0:
    print(f"\n{'='*75}")
    print("👻 DARK ART #4: PSEUDO-LABELING 👻")
    print(f"{'='*75}")

    for pl_round in range(1, PSEUDO_LABEL_ROUNDS + 1):
        print(f"\n━━━ Pseudo-Label Round {pl_round}/{PSEUDO_LABEL_ROUNDS} ━━━")

        # Yüksek güvenli tahminler
        high_conf_churn    = final_test_pred >= PL_HIGH_THRESHOLD  # Kesinlikle Churn
        high_conf_no_churn = final_test_pred <= PL_LOW_THRESHOLD   # Kesinlikle Churn değil
        mask = high_conf_churn | high_conf_no_churn

        n_pseudo = mask.sum()
        n_churn  = high_conf_churn.sum()
        n_no     = high_conf_no_churn.sum()
        print(f"   📊 Pseudo-label sayısı: {n_pseudo:,} (Churn={n_churn:,}, No={n_no:,})")
        print(f"   📊 Orijinal train: {len(y):,} → Genişletilmiş: {len(y) + n_pseudo:,}")

        if n_pseudo < 100:
            print(f"   ⚠️  Çok az pseudo-label ({n_pseudo}), round atlanıyor.")
            continue

        # Pseudo etiketler
        pseudo_labels = np.where(high_conf_churn[mask], 1.0, 0.0)

        # Genişletilmiş veri setleri oluştur (her model için kendi feature seti)
        X_xgb_ext  = np.vstack([X_xgb,  X_xgb_test[mask]])
        X_lgbm_ext = np.vstack([X_lgbm, X_lgbm_test[mask]])
        X_cat_ext  = np.vstack([X_cat,  X_cat_test[mask]])
        X_mlp_ext  = np.vstack([X_mlp,  X_mlp_test[mask]])
        y_ext      = np.concatenate([y, pseudo_labels])

        # Round 2 eğitimi — genişletilmiş veriyle
        (xgb_oof2, xgb_test2, _,
         lgbm_oof2, lgbm_test2, _,
         cat_oof2, cat_test2, _,
         mlp_oof2, mlp_test2, _) = run_full_training(
            X_xgb_ext, X_lgbm_ext, X_cat_ext, X_mlp_ext, y_ext,
            X_xgb_test, X_lgbm_test, X_cat_test, X_mlp_test,
            round_label=f"ROUND {pl_round + 1} (Pseudo-Labeled)"
        )

        # Sadece orijinal train kısmının OOF'unu al (pseudo kısmı hariç)
        n_orig = len(y)
        xgb_oof_orig  = xgb_oof2[:n_orig]
        lgbm_oof_orig = lgbm_oof2[:n_orig]
        cat_oof_orig  = cat_oof2[:n_orig]
        mlp_oof_orig  = mlp_oof2[:n_orig]

        # Tekrar Nelder-Mead
        def neg_auc_pl(weights):
            w = np.abs(weights)
            w = w / w.sum()
            blend = w[0]*xgb_oof_orig + w[1]*lgbm_oof_orig + w[2]*cat_oof_orig + w[3]*mlp_oof_orig
            return -roc_auc_score(y, blend)

        best_pl = None
        for init_w in [[0.25]*4, [0.35, 0.30, 0.25, 0.10], [0.30, 0.30, 0.30, 0.10]]:
            res_pl = minimize(neg_auc_pl, x0=init_w, method='Nelder-Mead',
                              options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
            if best_pl is None or res_pl.fun < best_pl.fun:
                best_pl = res_pl

        pl_w = np.abs(best_pl.x)
        pl_w = pl_w / pl_w.sum()
        pl_oof  = pl_w[0]*xgb_oof_orig + pl_w[1]*lgbm_oof_orig + pl_w[2]*cat_oof_orig + pl_w[3]*mlp_oof_orig
        pl_test = pl_w[0]*xgb_test2 + pl_w[1]*lgbm_test2 + pl_w[2]*cat_test2 + pl_w[3]*mlp_test2
        pl_auc  = roc_auc_score(y, pl_oof)

        print(f"\n   🔄 PL Round {pl_round} Ağırlıklar: XGB={pl_w[0]:.3f} LGBM={pl_w[1]:.3f} CAT={pl_w[2]:.3f} MLP={pl_w[3]:.3f}")
        print(f"   🎯 PL Round {pl_round} OOF AUC: {pl_auc:.5f} (önceki: {final_auc:.5f})")

        if pl_auc > final_auc:
            print(f"   ✅ İyileşme var! (+{pl_auc - final_auc:.6f})")
            final_test_pred = pl_test
            final_auc = pl_auc
        else:
            print(f"   ⚠️  İyileşme yok, önceki tahminler korunuyor.")

# ╔══════════════════════════════════════════════════════════════╗
# ║  7. SONUÇ RAPORU VE SUBMISSION                               ║
# ╚══════════════════════════════════════════════════════════════╝
print(f"\n{'='*75}")
print("📊 FİNAL SONUÇ RAPORU")
print(f"{'='*75}")
print(f"\n   Bireysel Model AUC'leri:")
print(f"     🎯 XGBoost  : {xgb_auc:.5f}  ({len(xgb_features)} feature)")
print(f"     🎯 LightGBM : {lgbm_auc:.5f}  ({len(lgbm_features)} feature)")
print(f"     🎯 CatBoost : {cat_auc:.5f}  ({len(cat_features)} feature)")
print(f"     🧠 MLP      : {mlp_auc:.5f}  ({len(mlp_features)} feature)")
print(f"   {'─'*50}")
print(f"   Ensemble Yöntemleri:")
print(f"     🔬 Nelder-Mead  : {nm_auc:.5f}")
print(f"     📚 Stacking     : {stack_auc:.5f}")
print(f"   {'─'*50}")
print(f"     🏆 FİNAL AUC    : {final_auc:.5f}  (Kazanan: {winner})")
if PSEUDO_LABEL_ROUNDS > 0:
    print(f"     👻 Pseudo-Label : {'Uygulandı' if final_auc > r1_auc else 'İyileşme sağlamadı'}")
print(f"{'='*75}")

# Submission dosyası
submission = pd.DataFrame({
    'id': test_ids,
    'Churn': final_test_pred
})
output_file = f'{OUTPUT_DIR}/submission_v5_darkarts.csv'
submission.to_csv(output_file, index=False)

elapsed = time.time() - t0
print(f"\n✅ Teslim dosyası: {output_file}")
print(f"⏱️  Toplam süre: {elapsed/60:.1f} dakika")
print(f"\n🏴‍☠️ Dark Arts Pipeline tamamlandı! Kaggle'a yükleyin! 🏴‍☠️")
