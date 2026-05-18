# script_ml.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# =====================
# Charger le dataset
# =====================
df = pd.read_csv("../DATA/dataset_synthetique.csv")

# Définir les cibles (t+1)
targets = [
    "ROA_t1", "ROE_t1", "NetMargin_t1",
    "OperatingMargin_t1", "CashRatio_t1", "OCG_t1"
]

# Définir scénarios de variables
scenarios = {
    "all variables": [col for col in df.columns if col not in targets],
    "base": ["Ratio_1", "Ratio_2", "Ratio_3"],  # exemple : à adapter
    "fin-st vars": [col for col in df.columns if "Ratio" in col],
    "new vars": ["NCG", "OCG", "CLCC", "OCS", "QPT", "QOFF", "LYCA", "IAICOC", "ROA2bond"]
}

# Définir modèles
models = {
    "Linreg": LinearRegression(),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
    "ElasNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "CART": DecisionTreeRegressor(max_depth=5),
    "SVR": SVR(kernel="rbf")
}

# Stocker résultats
results = []

# =====================
# Boucle principale
# =====================
for target in targets:
    for scen_name, features in scenarios.items():
        # Supprimer NA
        data = df[features + [target]].dropna()
        if data.empty:
            continue

        X = data[features]
        y = data[target]

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Normalisation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Tester chaque modèle
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                results.append([target, scen_name, model_name, mse])
            except Exception as e:
                print(f"Erreur avec {model_name} sur {
                      target}-{scen_name}: {e}")

# =====================
# Résultats
# =====================
results_df = pd.DataFrame(
    results, columns=["Predictive Task", "Feature Set", "Model", "MSE"])

# Transformer en tableau pivot 
pivot_df = results_df.pivot_table(
    index=["Predictive Task", "Feature Set"],
    columns="Model", values="MSE"
)

# Affichage joli avec couleurs
print("\n===== Résultats comparatifs MSE =====")
print(pivot_df)

#  rendu coloré dans Spyder (DataFrame stylisé)
try:
    from IPython.display import display
    display(pivot_df.style.background_gradient(cmap="RdYlGn_r"))
except:
    pass

# Sauvegarde dans un fichier CSV
pivot_df.to_csv("resultats_MSE.csv")
print("\nRésultats sauvegardés dans 'resultats_MSE.csv'")
df1 = pd.read_csv("resultats_MSE.csv")
