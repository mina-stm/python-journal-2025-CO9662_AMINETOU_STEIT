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
import matplotlib.pyplot as plt

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

# Exemple de données MSE (à remplacer par tes vrais résultats)
mse_data = {
    "ROA_t1": {"base": 0.00127, "fin_st_vars": 0.00129, "all_vars": 0.00132},
    "ROE_t1": {"base": 0.01742, "fin_st_vars": 0.01760, "all_vars": 0.01790},
    "NetMargin_t1": {"base": 0.05679, "fin_st_vars": 0.05695, "all_vars": 0.05710},
    "OpMargin_t1": {"base": 0.04816, "fin_st_vars": 0.04850, "all_vars": 0.04900},
    "CashRatio_t1": {"base": 0.27908, "fin_st_vars": 0.28000, "all_vars": 0.27800},
    "OCG_t1": {"base": 6.9191, "fin_st_vars": 6.8500, "all_vars": 6.8000}
}

tasks = list(mse_data.keys())
scenarios = ["base", "fin_st_vars", "all_vars"]

# Indexation par rapport au scénario base
indexed = {scenario: [] for scenario in scenarios}
for task in tasks:
    base_mse = mse_data[task]["base"]
    for scenario in scenarios:
        value = mse_data[task][scenario]
        index = (value / base_mse) * 100
        indexed[scenario].append(index)

# Tracé de la figure
plt.figure(figsize=(10,6))
for scenario, values in indexed.items():
    plt.plot(tasks, values, marker='o', label=scenario)

plt.axhline(100, color='red', linestyle='--', label="Base = 100")
plt.title("Régression Linéaire – MSE indexée (base = 100)")
plt.ylabel("Index MSE (%)")
plt.xlabel("Tâches prédictives")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

