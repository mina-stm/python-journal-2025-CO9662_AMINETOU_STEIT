import pandas as pd # Importing Pandas
import matplotlib.pyplot as plt # Importing matplotlib for visualisation
df = pd.read_csv('../Data/pokemon_data.csv')
print(df.iloc[2, 4])   # Here we can see the information from second row and fourth␣column
print(df.iloc[:5, :5]) # Here is the information of first five rows with first five␣↪columns
df.iloc[:5, 1:5].plot.bar() # Bar char of the DataFrame provided above
plt.show()
print(df.iloc[5])
print(df.iloc[[5]])
print(df.iloc[:, 1])
print(df.iloc[:, [1, 6]]) # Filters all rows and "Name" and "Defense" columns as␣↪DataFrame
df.iloc[:, [1, 6]].plot.hist()
df['Legendary'] = df['special_group'] == 'Legendary'
print(df.loc[df['Legendary'] == False])
print(df.columns)
print(df.loc[df['special_group'] != 'Legendary'])
print(df.loc[
    (df['Legendary'] == False)
    & (df['type1'] == 'Grass')
])
print(df.columns)
print(df.loc[
    ~((df['type1'] == 'Grass')
    & (df['Legendary'] == False))
])
print(df.columns)
print(df.query('(hp > 40) and (attack < 100)'))
min_hp = 70
print(df.query('(hp > @min_hp)'))
