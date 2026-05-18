# ================================
# Script Hybride Réseau Bayésien
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer

from pgmpy.estimators import HillClimbSearch, BDeu
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# ================================
# 1. Charger et préparer les données
# ================================
df5 = pd.read_csv("../DATA/dataset_synthetique.csv")

# Discrétisation (adapter n_bins si besoin)
disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
df_disc = pd.DataFrame(disc.fit_transform(df5), columns=df5.columns)

print("\n--- Vérification après discrétisation ---")
for col in df_disc.columns:
    print(col, ":", df_disc[col].unique())

# ================================
# 2. Recherche de structure automatique
# ================================
hc = HillClimbSearch(df_disc)
best_model = hc.estimate(scoring_method=BDeu(df_disc))

print("\nArêtes trouvées automatiquement :", list(best_model.edges()))

# ================================
# 3. Si modèle vide → définir un modèle manuel
# ================================
if len(best_model.edges()) == 0:
    print("\n⚠️ Aucun lien trouvé → Construction d’un modèle par défaut.")
    model = DiscreteBayesianNetwork([
        ("GDP_growth", "NetMargin_t1"),
        ("Inflation", "NetMargin_t1"),
        ("Unemployment", "NetMargin_t1")
    ])
else:
    model = DiscreteBayesianNetwork(best_model.edges())

# ================================
# 4. Apprentissage des CPDs
# ================================
model.fit(df_disc, estimator=BayesianEstimator, prior_type="BDeu")

print("\nCPDs apprises :")
for cpd in model.get_cpds():
    print(cpd)

# ================================
# 5. Visualisation du graphe
# ================================


# 5. Visualisation du graphe
plt.figure(figsize=(8,6))

# Construire un graphe networkx à partir du modèle
graph = nx.DiGraph(model.edges())

nx.draw(graph, with_labels=True, node_size=2000, 
        node_color="lightblue", font_size=10, arrows=True)

plt.title("Réseau Bayésien")
plt.show(block=True)




# ================================
# 6. Inférence
# ================================
inference = VariableElimination(model)

try:
    query = inference.query(variables=["NetMargin_t1"], evidence={"GDP_growth": 2})
    print("\nDistribution prédite de NetMargin_t1 si GDP_growth=2 :")
    print(query)
except Exception as e:
    print("\n⚠️ Impossible d’effectuer l’inférence :", e)
