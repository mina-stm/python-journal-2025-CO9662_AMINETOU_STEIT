# ======================================================
# Script complet - Méthodologie article appliquée à dataset_synthetique.csv
# ======================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ======================================================
# 1. Charger le dataset
# ======================================================
df = pd.read_csv("../data/dataset_synthetique.csv")
df
print(df.columns)
# ======================================================
# 2. Définir les groupes de variables
# ======================================================
ratios_BS_IS = [c for c in df.columns if c.startswith("BS_") or c.startswith("IS_")]
ratios_CF = [c for c in df.columns if c.startswith("CF_")]
ratios_MRFS = [c for c in df.columns if c.startswith("MRFS_")]
macro_idx = [c for c in df.columns if c.startswith("MACRO_")]

targets = ["ROA_t1", "ROE_t1", "NetMargin_t1", "OperatingMargin_t1", "CashRatio_t1", "OCG_t1"]
# targets = ["ROA_t1", "ROE_t1", "NetMargin_t1", "OperatingMargin_t1", "CashRatio_t1", "OCG_t1"]

# ======================================================
# 3. Définir les scénarios de variables
# ======================================================
scenarios = {
    "base": [f"Ratio_{i}" for i in range(1, 25)],  # 24 ratios BS + IS
    "all": [col for col in df.columns if col not in targets],  # toutes les colonnes sauf cibles
    "fin-st_vars": [f"Ratio_{i}" for i in range(1, 25)] + ["NCG", "OCG", "CLCC", "OCS", "QPT", "QOFF", "LYCA", "IAICOC", "ROA2bond"],
    "new_vars": ["NCG", "OCG", "CLCC", "OCS", "QPT", "QOFF", "LYCA", "IAICOC", "ROA2bond"]
}

#scenarios = {
#    "base": ratios_BS_IS,  # 24 ratios BS + IS
#   "all_vars": ratios_BS_IS + ratios_CF + ratios_MRFS + macro_idx,  # 43 + macro
#    "fin_st_vars": ratios_BS_IS + ratios_CF + ratios_MRFS,  # 33 ratios
#    "new_vars": ratios_CF + ratios_MRFS  # 9 nouveaux ratios
#}

# ======================================================
# 4. Prétraitement
# ======================================================
scaler = StandardScaler()

# ======================================================
# 5. Modèles ML
# ======================================================
models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1)
}

# Réseau de neurones (FNN simplifié)
def build_fnn(input_dim, deep=False, wide=False):
    if deep and wide:
        hidden_layer_sizes = (100, 20)
    elif deep:
        hidden_layer_sizes = (input_dim+20, input_dim)
    elif wide:
        hidden_layer_sizes = (100,)
    else:
        hidden_layer_sizes = (input_dim,)
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="logistic",
        solver="adam",
        max_iter=1000
    )

# ======================================================
# 6. Expérimentation
# ======================================================
results = {}

for scen, variables in scenarios.items():
    print(f"\n==============================")
    print(f" Scénario : {scen} ({len(variables)} variables)")
    print(f"==============================")

    X = df[variables].fillna(0)
    y = df[targets]

    # Normalisation
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results[scen] = {}

    for target in targets:
        results[scen][target] = {}
        y_train_target = y_train[target]
        y_test_target = y_test[target]

        # --- Modèles ML
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train_target, cv=10, scoring="r2")
            model.fit(X_train, y_train_target)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test_target, y_pred)
            r2 = r2_score(y_test_target, y_pred)
            results[scen][target][name] = {"CV_R2": scores.mean(), "Test_R2": r2, "MSE": mse}

        # --- Réseau de neurones
        fnn_base = build_fnn(X_train.shape[1])
        fnn_base.fit(X_train, y_train_target)
        y_pred_fnn = fnn_base.predict(X_test)
        r2_fnn = r2_score(y_test_target, y_pred_fnn)
        results[scen][target]["FNN_base"] = {"Test_R2": r2_fnn}

print("\nExpérimentation terminée ✅")
