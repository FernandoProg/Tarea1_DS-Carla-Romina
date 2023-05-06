import pandas as pd
import numpy as np

pokemon = pd.read_csv('./datasets/pokemon.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
pokemon.index += 1   # Los pokemon son enumerados del 1 al 800 en vez del 0 al 799

# # 2.1 Perfilado

# # 1. Cantidad de atributos
# print('En el dataset hay ' + str(pokemon.shape[1]) + ' atributos')
# # 2. Cantidad de registros
# print('En el dataset hay ' + str(pokemon.shape[0]) + ' registros')
# # Ademas, entregar la descripcion completa de cada atributo dataset incluyendo:
# print('Tipo de dato:\n' + str(pokemon.dtypes))
# print('Valores faltantes:\n' + str(pokemon[pokemon['Type 2'].isna()]))
# print('Minimo (si es numerico):\n' + str(pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].min(axis=0, numeric_only=True, skipna=True)))
# print('Maximo (si es numerico):\n' + str(pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].max(axis=0, numeric_only=True, skipna=True)))
# print('Desviacion estandar (si es numerico):\n' + str(pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].std(axis=0, numeric_only=True, skipna=True)))

# # 2.2 Preprocesamiento

# # 1. Normalizacion de todas las variables numericas. Indicando claramente que tipo de reescalamiento o normalizacion aplico y porque.
norm_pokemon = (pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']] - pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].min()) / ( pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].max() - pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].min())
norm_pokemon = pd.concat([norm_pokemon, pokemon.loc[:, ['Name', 'Type 1', 'Type 2']]], axis=1)[['Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
# print(norm_pokemon)
# # 2. Discretizacion de todas las variables numericas:
# ?

# # Valores Nulos y Outliers
pokemon_with_nulls = pd.read_csv('./datasets/pokemon_with_nulls.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
pokemon_with_nulls.index += 1   # Los pokemon son enumerados del 1 al 800 en vez del 0 al 799
print(pokemon_with_nulls)
