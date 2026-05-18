# 1 Notions de base sur les DataFrames
import pandas as pd
import matplotlib.pyplot as plt
data = {'Nom': ['Alice', 'Bob'], 'Âge': [25, 30]}
df0 = pd.DataFrame(data) # créer un dataframe a partir d'un dictionnaire
df0 = pd.read_csv('../data/dataset_synthetique1.csv') # Lire et écrire des fichiers CSV/Excel
df0
df0.shape      # (lignes, colonnes)
df0.size       # total d’éléments
df0.dtypes     # types des colonnes
df0.info()     # résumé complet
print(df0.head())  # premières lignes
print(df0.tail())  # dernieres lignes
df0.rename(columns={'Ratio_1':'Ratio1'},inplace=True) # Renommer des colonnes
df0.reset_index(drop=True, inplace=True)     # réinitialiser
#df0.set_index('Ratio1', inplace=True)           # définir 'Nom' comme index
# df = pd.read_excel('fichier.xlsx')
# df.to_excel('nouveau.xlsx', index=False) Lire et écrire des fichiers CSV/Excel
# 2 lecture et écriture des fichiers CSV/Excel
# df = pd.read_csv('../data/pokemon_data.csv') # Lire  des fichiers CSV
# df = pd.read_excel('../data/pokemon_data.xlsx') # Lire  des fichiers excel
# df.to_csv('sortie.csv', index=False) écriture des fichiers CSV
# df.to_excel('sortie.xlsx', index=False) écriture des fichiers Excel
# Gestion des données Catégorielles
# df['colonne'] = df['colonne'].astype('category')
### Q8. Détecter et gérer les données manquantes python
print(df0.isnull().sum())    # détection
# df.fillna(valeur)       # remplissage
# df.dropna()             # suppression
### Q9. Détecter et supprimer les doublons
# df.drop_duplicates(inplace=True)
### Q10. Remplacer des valeurs spécifiques
# df0.replace({'ancien': 'nouveau'}, inplace=True)
### Q11. Trier par colonnes
# print(df0.sort_values(by='type1', ascending=False))
### Q12. Réinitialiser l’index
#df.reset_index(drop=True)


### Q13. Sélectionner des lignes selon des conditions

#print(df0[df0['hp'] == '45'])
import pgmpy
import pgmpy.estimators as est
print(dir(est))