# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:19:32 2026

@author: PC
"""
import pandas as pd
import matplotlib.pyplot as plt

# Données MSE du Tableau 3
data = {
    "Predictive Task": ["ROA", "ROE", "Net Margin", "Op. Margin", "Cash Ratio", "OCG"],
    "Linear Regr.": [0.0013, 0.0174, 0.0569, 0.0483, 0.2802, 7.2926],
    "AdaB": [0.0020, 0.0451, 0.0679, 0.1762, 0.2576, 24.3549],
    "GradB": [0.0012, 0.0167, 0.0546, 0.0456, 0.2835, 6.2345],
    "RF": [0.0014, 0.0187, 0.0604, 0.0501, 0.3230, 6.6726],
    "ET": [0.0014, 0.0183, 0.0596, 0.0506, 0.3172, 6.6480],
    "HistGB": [0.0012, 0.0167, 0.0542, 0.0454, 0.2817, 6.2149],
    "Vote": [0.0013, 0.0172, 0.0582, 0.0481, 0.3282, 6.4670]
}

df3 = pd.DataFrame(data)
print(df3)




# Données MSE du tableau 3 (scénario fin-st vars)
mse_data = {
    "ROA_t1": {"Linear": 0.0013, "AdaB": 0.0020, "GradB": 0.0012, "RF": 0.0014, "ET": 0.0014, "HistGB": 0.0012, "Vote": 0.0013},
    "ROE_t1": {"Linear": 0.0174, "AdaB": 0.0451, "GradB": 0.0167, "RF": 0.0187, "ET": 0.0183, "HistGB": 0.0167, "Vote": 0.0172},
    "NetMargin_t1": {"Linear": 0.0569, "AdaB": 0.0679, "GradB": 0.0546, "RF": 0.0604, "ET": 0.0596, "HistGB": 0.0542, "Vote": 0.0582},
    "OpMargin_t1": {"Linear": 0.0483, "AdaB": 0.1762, "GradB": 0.0456, "RF": 0.0501, "ET": 0.0506, "HistGB": 0.0454, "Vote": 0.0481},
    "CashRatio_t1": {"Linear": 0.2802, "AdaB": 0.2576, "GradB": 0.2835, "RF": 0.3230, "ET": 0.3172, "HistGB": 0.2817, "Vote": 0.3282},
    "OCG_t1": {"Linear": 7.2926, "AdaB": 24.3549, "GradB": 6.2345, "RF": 6.6726, "ET": 6.6480, "HistGB": 6.2149, "Vote": 6.4670}
}

tasks = list(mse_data.keys())
models = ["Linear", "AdaB", "GradB", "RF", "ET", "HistGB", "Vote"]

# Indexation par rapport à Linear Regression (base = 100)
indexed = {model: [] for model in models}
for task in tasks:
    base = mse_data[task]["Linear"]
    for model in models:
        value = mse_data[task][model]
        index = (value / base) * 100
        indexed[model].append(index)

# Tracé de la figure
plt.figure(figsize=(12,6))
for model in models:
    plt.plot(tasks, indexed[model], marker='o', label=model)

plt.axhline(100, color='red', linestyle='--', label="Régression linéaire = 100")
plt.title("Performance des modèles d’ensemble – MSE indexée (base = Régression linéaire)")
plt.ylabel("Index MSE (%)")
plt.xlabel("Tâches prédictives")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
