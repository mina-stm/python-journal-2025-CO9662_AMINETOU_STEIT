# script_causal_step_by_step.py
# Exécuter depuis Code/ avec dataset dans ../Data/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer

# ---- pgmpy imports robustes selon version ----
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import HillClimbSearch, BayesianEstimator
    from pgmpy.inference import VariableElimination
except Exception:
    # tentatives fallback
    try:
        from pgmpy.models import BayesianModel as DiscreteBayesianNetwork
        from pgmpy.estimators import HillClimbSearch, BayesianEstimator
        from pgmpy.inference import VariableElimination
    except Exception as e:
        raise ImportError("pgmpy non disponible ou incompatible. Installe/upgrade pgmpy.") from e

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "../Data/dataset_synthetique.csv"   # adapte si besoin
# Choisis un sous-ensemble raisonnable de variables (macro + cibles)
vars_to_use = [
    "Unemployment", "GDP_growth", "Inflation", "Interest_rate", "Bond10Y",
    "ROA_t1", "ROE_t1", "NetMargin_t1", "OperatingMargin_t1", "CashRatio_t1", "OCG_t1"
]
N_BINS = 3   # nombre de catégories pour discrétisation
RANDOM_STATE = 42

# ---------------------------
# 1) Charger et afficher colonnes
# ---------------------------
df4 = pd.read_csv(DATA_PATH)
print("Colonnes disponibles dans le CSV :")
print(df4.columns.tolist())

# Garder uniquement les colonnes demandées (filtre si certaines manquent)
vars_present = [c for c in vars_to_use if c in df4.columns]
missing = [c for c in vars_to_use if c not in vars_present]
if missing:
    print("⚠️ Ces variables n'existent pas dans le dataset et seront ignorées :", missing)
print("Variables retenues pour le BN :", vars_present)

# Extraire le sous-DataFrame
df_sub = df4[vars_present].copy()

# ---------------------------
# 2) Discrétisation (toutes les colonnes doivent être discrètes pour pgmpy)
# ---------------------------
print("\n→ Discrétisation en cours (KBins, strategy='quantile') ...")
kbd = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
# KBinsDiscretizer peut échouer sur colonnes non numériques -> forcer float
numeric_cols = df_sub.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) != len(df_sub.columns):
    print("⚠️ Certaines colonnes ne sont pas numériques, je tente de les convertir.")
    for c in df_sub.columns:
        df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')

# Drop lignes contenant NaN (après conversion / si beaucoup, tu peux imputer)
df_sub = df_sub.dropna()
if df_sub.empty:
    raise ValueError("Après nettoyage/discrétisation, le DataFrame est vide. Vérifie les données.")

# Appliquer KBinsDiscretizer colonne par colonne (plus robuste que .fit_transform sur tout)
df_disc = pd.DataFrame(index=df_sub.index)
for col in df_sub.columns:
    arr = df_sub[[col]].values
    # si toutes valeurs identiques -> faire une colonne 0
    if np.allclose(arr, arr[0]):
        df_disc[col] = 0
        continue
    try:
        kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
        res = kbd.fit_transform(arr)
        df_disc[col] = res.astype(int).ravel()
    except Exception as e:
        # fallback : cut personnalisé en 3 bins
        df_disc[col] = pd.cut(df_sub[col], bins=N_BINS, labels=False, duplicates='drop').astype(int)
print("Discrétisation terminée. Aperçu :")
print(df_disc.head())
print("\n--- Vérification des variables après discrétisation ---")
for col in df_disc.columns:
    print(col, ":", df_disc[col].unique())

# ---------------------------
# 3) Apprentissage de la structure (Hill-Climb)
# ---------------------------
print("\n→ Apprentissage de la structure (HillClimbSearch) ...")
hc = HillClimbSearch(df_disc)
try:
    best = hc.estimate()   # si ta version supporte scoring_method tu peux l'ajouter
except Exception as e:
    # fallback plus sûr : heuristique simple en limitant l'espace
    print("Warning: HillClimbSearch.estimate() a levé:", e)
    best = hc.estimate(max_indegree=2)  # ou ajuster max_indegree

print("Arêtes trouvées :", list(best.edges()))
# Afficher noeuds retenus
print("Noeuds du graphe appris :", list(best.nodes()))




# ---------------------------
# 4) Construire le modèle et apprendre les CPDs
# ---------------------------


print("\n→ Construction du DiscreteBayesianNetwork et apprentissage des CPDs ...")
model = DiscreteBayesianNetwork(list(best.edges()))
# fit (BayesianEstimator) — si erreur, essayer MaximumLikelihoodEstimator
try:
    model.fit(df_disc, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
except Exception as e:
    print("BayesianEstimator a échoué (fallback à MLE) :", e)
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model.fit(df_disc, estimator=MaximumLikelihoodEstimator)

# Vérifier CPDs
print("\nCPDs apprises (liste) :")
for cpd in model.get_cpds():
    print(cpd)

# ---------------------------
# 5) Afficher la structure (graph) et attendre validation
# ---------------------------
# 5. Visualisation du graphe
plt.figure(figsize=(8,6))

# Construire un graphe networkx à partir du modèle
graph = nx.DiGraph(model.edges())

nx.draw(graph, with_labels=True, node_size=2000, 
        node_color="lightblue", font_size=10, arrows=True)

plt.title("Réseau Bayésien")
plt.show(block=True)


# ---------------------------
# 6) Afficher chaque CPD (texte + graphique) un par un
# ---------------------------
from pgmpy.factors.discrete import TabularCPD

def plot_cpd(cpd):
    # cpd peut être TabularCPD ou similaire ; essaye de récupérer valeurs
    try:
        vals = np.array(cpd.get_values())  # shape (cardinality_of_var, product_parent_cards)
    except Exception:
        try:
            vals = np.array(cpd.values)
        except Exception:
            print("Impossible d'extraire valeurs du CPD pour", cpd.variable)
            return
    # Si pas de parents -> vecteur de probs
    if vals.ndim == 1 or vals.shape[1] == 1:
        v = vals.ravel()
        plt.figure()
        plt.bar(range(len(v)), v)
        plt.xlabel("Etat")
        plt.ylabel("Probabilité")
        plt.title(f"CPD: {cpd.variable} (marginal)")
        plt.xticks(range(len(v)))
        plt.show()
    else:
        # matrice : rows = child states, cols = parent-configs
        plt.figure(figsize=(6,4))
        plt.imshow(vals, aspect='auto')
        plt.colorbar(label='Prob')
        plt.xlabel("Configuration parents")
        plt.ylabel("Etat enfant")
        plt.title(f"CPD: {cpd.variable} (cond. sur parents)")
        plt.show()

for cpd in model.get_cpds():
    print("\n--- CPD pour :", cpd.variable, "---")
    print(cpd)
    plot_cpd(cpd)
    input(">> CPD affiché. Appuie sur Entrée pour suivre ...")

# ---------------------------
# 7) Inférence : exemple (VariableElimination)
# ---------------------------
inference = VariableElimination(model)
print("\nNoeuds du modèle:", list(model.nodes()))

# Choisir une evidence variable qui existe dans le modèle
# Exemple: si 'GDP_growth' est dans model.nodes() l'utiliser sinon en choisir une présente
evidence_var = None
for v in ["GDP_growth", "Inflation", "Interest_rate", "Unemployment"]:
    if v in model.nodes():
        evidence_var = v
        break

if evidence_var is None:
    print("Aucune variable d'évidence macro trouvée dans le graphe. Liste des noeuds:", list(model.nodes()))
else:
    ev_value = 1  # valeur de l'état discret (0..N_BINS-1) — change selon ce qui te paraît logique
    print(f"\n→ Exemple d'inférence: P(NetMargin_t1 | {evidence_var}={ev_value})")
    if "NetMargin_t1" not in model.nodes():
        print("Target 'NetMargin_t1' non dans le graphe — choisis une cible différente.")
    else:
        q = inference.query(variables=["NetMargin_t1"], evidence={evidence_var: ev_value})
        print(q)
        # afficher distribution comme bar plot
        cpd_vals = np.array(q.values).ravel()
        plt.figure()
        plt.bar(range(len(cpd_vals)), cpd_vals)
        plt.title(f"P(NetMargin_t1 | {evidence_var}={ev_value})")
        plt.xlabel("Etat NetMargin_t1")
        plt.ylabel("Probabilité")
        plt.show()

print("\n== Fin du script ==")
