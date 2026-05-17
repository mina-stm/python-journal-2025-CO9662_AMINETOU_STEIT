import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Chargement du dataset
df5 = pd.read_csv("../DATA/dataset_synthetique.csv")
df5.dropna(inplace=True)

# Définition des scénarios
base_vars  = [f"Ratio_{i}" for i in range(1, 25)]
cf_mrfs_vars = ["NCG","OCG","CLCC","OCS","QPT","QOFF","LYCA","IAICOC","ROA2bond"]
macro_vars = ["Unemployment","GDP_growth","Inflation","Bond10Y","Interest_rate",
              "Exchange_rate","Commodity_index","Consumer_sentiment","Fiscal_balance","Trade_balance"]

scenarios = {
    "base": base_vars,
    "fin-st vars": base_vars + cf_mrfs_vars,
    "all variables": base_vars + cf_mrfs_vars + macro_vars,
    "new vars": cf_mrfs_vars
}

# Définir les cibles
targets = ["ROA_t1","ROE_t1","NetMargin_t1","OperatingMargin_t1","CashRatio_t1","OCG_t1"]

# Définition des architectures
architectures = {
    "NN Base": [20],
    "NN Deep": [100, 20],
    "NN Wide": [100],
    "NN Deep & Wide": [100, 20]
}

results = []

# Boucle principale
for target in targets:
    for scenario_name, scenario_features in scenarios.items():
        X = df5[scenario_features].select_dtypes(include=[np.number])
        y = df5[target]

        if X.empty:
            print(f"⚠️ Scénario '{scenario_name}' ignoré pour {target} (aucune variable numérique).")
            continue

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        for model_name, layers in architectures.items():
            model = Sequential()
            model.add(Dense(layers[0], activation='sigmoid', input_shape=(X_train.shape[1],)))
            if len(layers) > 1:
                model.add(Dense(layers[1], activation='sigmoid'))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            results.append([target, scenario_name, model_name, round(mse, 6)])

# Résultats en DataFrame
results_df1 = pd.DataFrame(results, columns=["Predictive Task","Scenario","Model","MSE"])
print(results_df1)
