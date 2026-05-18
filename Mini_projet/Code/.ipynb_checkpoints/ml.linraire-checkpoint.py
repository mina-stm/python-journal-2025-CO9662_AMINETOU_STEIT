# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 01:32:55 2026

@author: PC
"""

# 1. Importer les bibliothèques
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# 2. Charger ta base de données
df = pd.read_csv("../DATA/dataset_synthetique.csv")  # adapte le nom du fichier

# 3. Définir les variables
features = df.drop(columns=["ROA_t1", "ROE_t1", "NetMargin_t1", "OperatingMargin_t1", "CashRatio_t1", "OCG_t1"])  # variables explicatives
# targets = ["ROA", "ROE", "NetMargin", "OpMargin", "CashRatio", "OpCashGen"]  # variables cibles
targets = [
    "ROA_t1", "ROE_t1", "NetMargin_t1",
    "OperatingMargin_t1", "CashRatio_t1", "OCG_t1"
]
# 4. Définir les modèles
models = {
    "LinearRegression": LinearRegression(),
    "SVR": SVR()
}

# 5. Évaluer chaque modèle pour chaque tâche
results = []

for target in targets:
    y = df[target].values
    X = features.values
    for model_name, model in models.items():
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        avg_mse = np.mean(mse_scores)
        results.append({
            "Task": target,
            "Model": model_name,
            "MSE": round(avg_mse, 5)
        })

# 6. Afficher les résultats
results_df0 = pd.DataFrame(results)
print(results_df0.sort_values(by="MSE"))
