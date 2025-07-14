import pandas as pd # Importing pandas
import matplotlib.pyplot as plt
df = pd.read_csv('../Data/pokemon_data.csv')
df
print(df.head())         # Affiche les 5 premières lignes
df.head().plot.bar()
print(df.head(10))       # Affiche les 10 premières lignes
print(df.tail(10))       # Affiche les 10 dernières lignes
df.info()
df.info(verbose=False)
print(df.sample(5))
print(df.sample(frac = 0.1))
print(df.columns)
print(df.index)
print(df.describe())
print(df[['hp', 'attack']].describe())
print(df[['hp']]) # Shows the DataFrame of "HP" column
print(df['hp']) # Shows the DataFrame of "HP" column
df['hp'].plot.hist()
plt.show() # Shows the histogram of DataFrame in "HP" column
print(df[df.columns[:4]]) 
print(df.select_dtypes('int')) # here we see rows only with filtering columns as intege