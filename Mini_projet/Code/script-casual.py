import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.estimators import HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator

# ===========================
# 1. Charger le dataset
# ===========================
df2 = pd.read_csv("../data/dataset_synthetique.csv")

print("Aperçu des données :")
print(df2.head())
df2.info()     # résumé complet

# ===========================
# 2. Apprentissage de la structure
# ===========================
hc = HillClimbSearch(df2)
best_model = hc.estimate()

print("Arêtes trouvées :", best_model.edges())

# Construire le modèle
model = DiscreteBayesianNetwork(best_model.edges())

# ===========================
# 3. Apprentissage des CPDs (distributions conditionnelles)
# ===========================
model.fit(df2, estimator=BayesianEstimator, prior_type="BDeu")

# ===========================
# 4. Visualiser le graphe
# ===========================
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(model, seed=42)  # layout graphique
nx.draw(
    model,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    arrowsize=20,
    font_weight="bold"
)
plt.title("Causal Bayesian Network appris")
plt.show()


from sklearn.preprocessing import KBinsDiscretizer
model.add_edge("GDP_growth", "NetMargin_t1")
# Exemple : suppose que ton DataFrame s’appelle df
cols_to_discretize = df2.columns  # toutes les colonnes
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

df_discrete = pd.DataFrame(discretizer.fit_transform(df2[cols_to_discretize]),
                           columns=cols_to_discretize)

df_discrete = df_discrete.astype(int)  # pgmpy attend des entiers (0,1,2,...)

print(df_discrete.head())

model.fit(df_discrete, estimator=BayesianEstimator, prior_type="BDeu")
print("Colonnes du DataFrame :", df_discrete.columns.tolist())
print("Nœuds du modèle :", list(model.nodes()))
# ===========================
# 5. Exemple d'inférence
# ===========================
from pgmpy.inference import VariableElimination

inference = VariableElimination(model)
print("Variables présentes dans le graphe :", model.nodes())
# Exemple : probabilité de Net Margin en fonction du PIB
query = inference.query(variables=["NetMargin_t1"], evidence={"GDP_growth": 2})
print("\nDistribution prédite de NetMargin_t1 :")
print(query)
