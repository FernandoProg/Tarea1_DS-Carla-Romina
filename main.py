import pandas as pd # pip install pandas
import numpy as np  # pip install numpy
from sklearn.preprocessing import KBinsDiscretizer  # pip install -U scikit-learn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from matplotlib.colors import ListedColormap
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

pokemon = pd.read_csv('./datasets/pokemon.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
pokemon.index += 1   # Los pokemon son enumerados del 1 al 800 en vez del 0 al 799

# 2.1 Perfilado

# 1. Cantidad de atributos
print('En el dataset hay ' + str(pokemon.shape[1]) + ' atributos')
# 2. Cantidad de registros
print('En el dataset hay ' + str(pokemon.shape[0]) + ' registros')
# Ademas, entregar la descripcion completa de cada atributo dataset incluyendo:
print('Tipo de dato:\n' + str(pokemon.dtypes))
print('Valores faltantes:\n' + str(pokemon[pokemon['Type 2'].isna()]))  # Con el comando print(pokemon.isnull().sum()) podemos ver que solo Type 2 tiene valores NaN
print('Minimo (si es numerico):\n' + str(pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].min(axis=0, numeric_only=True, skipna=True)))
print('Maximo (si es numerico):\n' + str(pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].max(axis=0, numeric_only=True, skipna=True)))
print('Desviacion estandar (si es numerico):\n' + str(pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].std(axis=0, numeric_only=True, skipna=True)))

# # 2.2 Preprocesamiento

# 1. Normalizacion de todas las variables numericas. Indicando claramente que tipo de reescalamiento o normalizacion aplico y porque.
norm_pokemon = (pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']] - pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].min()) / ( pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].max() - pokemon.loc[:, ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].min())
norm_pokemon = pd.concat([norm_pokemon, pokemon.loc[:, ['Name', 'Type 1', 'Type 2']]], axis=1)[['Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]

# 2. Discretizacion de todas las variables numericas:
# Definimos los objetos para las transformaciones
kbins_equal_width = KBinsDiscretizer(
    n_bins = 5,
    encode = 'ordinal',
    strategy = 'uniform'
)

kbins_equal_features = KBinsDiscretizer(
    n_bins = 5,
    encode = 'ordinal',
    strategy = 'quantile'
)

kbins_ew = pd.DataFrame(
    data = kbins_equal_width.fit_transform(norm_pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]),
    columns = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
)

kbins_ef = pd.DataFrame(
    data = kbins_equal_features.fit_transform(norm_pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]),
    columns = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
)

bins_edges = kbins_equal_width.bin_edges_[0]
norm_pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].hist(bins = bins_edges, edgecolor='k', linewidth=2, zorder=2)
for edge in bins_edges:
  plt.axvline(x=edge, color='r', linestyle='--', linewidth=2, alpha=0.75)
plt.show()

bins_edges = kbins_equal_features.bin_edges_[0]
norm_pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].hist(bins = bins_edges, edgecolor='k', linewidth=2, zorder=2)
for edge in bins_edges:
  plt.axvline(x=edge, color='r', linestyle='--', linewidth=2, alpha=0.75)
plt.show()

# Elija alguna variable categorica del dataset y conviertala a numerica. Porque aplico la transformacion a esta columna?
ordinal_encoder = OrdinalEncoder()
norm_pokemon['Type_OE'] = ordinal_encoder.fit_transform(norm_pokemon[['Type 1']])

# Valores Nulos y Outliers
pokemon_with_nulls = pd.read_csv('./datasets/pokemon_with_nulls.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
pokemon_with_nulls.index += 1   # Los pokemon son enumerados del 1 al 800 en vez del 0 al 799

# 1. Aplique imputacion a los datos para rellenar los nulos. Defina claramente cual imputacion aplico y porque.
pokemon_with_nulls['Type_OE'] = ordinal_encoder.fit_transform(pokemon_with_nulls[['Type 1']])
cols = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']
imputed_cols = [f'{c}_imp' for c in cols]
imputer = KNNImputer()
imputed_cols = [f'{c}_imp' for c in cols]
data_imputed = pd.DataFrame(
    data=imputer.fit_transform(pokemon_with_nulls[cols]),
    columns=imputed_cols
)
pokemon_with_nulls_imp = pd.concat([pokemon_with_nulls, data_imputed], axis=1)

# 2. Compare los resultados de la imputacion sobre el dataset pokemon_with_nulls y el dataset original pokemon. Concluya que tan efectiva fue la imputacion de los datos.
colors = ['red', 'cyan', 'green', 'orange', 'purple']
cmap = ListedColormap(colors)
fig, ax = plt.subplots()
ax.scatter(pokemon_with_nulls_imp['Attack'], pokemon_with_nulls_imp['Defense'], c='blue', marker='o', label='Attack-Defense',alpha=0.3, edgecolors='none')
ax.scatter(pokemon_with_nulls_imp[pokemon_with_nulls_imp['Attack'].isnull() | pokemon_with_nulls_imp['Defense'].isnull()]['Attack_imp'], 
           pokemon_with_nulls_imp[pokemon_with_nulls_imp['Attack'].isnull() | pokemon_with_nulls_imp['Defense'].isnull()]['Defense_imp'],
           c=pokemon_with_nulls_imp[pokemon_with_nulls_imp['Attack'].isnull() | pokemon_with_nulls_imp['Defense'].isnull()]['Type_OE'], 
           cmap=cmap, marker='*', label='Datos Imputados')
medias = pokemon_with_nulls_imp[['Attack', 'Defense']].mean()
ax.axhline(y=medias['Defense'], color='green', linestyle='--', linewidth=1, alpha=0.6)
ax.axvline(x=medias['Attack'], color='green', linestyle='--', linewidth=1, alpha=0.6)
ax.set_xlabel('Attack')
ax.set_ylabel('Defense')
ax.set_title('Gráfico de dispersión')
ax.grid(True)
ax.legend(loc='upper left')
plt.show()

# Para los valores atipicos utilice el dataset pokemon
# 1. Obtenga mediante candidatos a Outliers aplicando los metodos vistos en clases.
X = pokemon.drop(['Name', 'Type 1', 'Type 2'], axis=1)
y = pokemon[['Name', 'Type 1']]

# Detecta valores atípicos usando LOF
lof = LocalOutlierFactor(n_neighbors=20)
y_pred_lof = lof.fit_predict(X)
mask_lof = y_pred_lof == -1

# Detecta valores atípicos usando KNN
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)
y_pred_knn = knn.predict(X)
mask_knn = (y != y_pred_knn)

# Crea el gráfico de dispersión para LOF
fig, ax = plt.subplots()
ax.scatter(X['Attack'], X['Defense'], marker='o', label='Datos', alpha=0.3, edgecolors='none')
ax.scatter(X[mask_lof]['Attack'], X[mask_lof]['Defense'], marker='*', label='LOF outliers')
ax.set_xlabel('Attack')
ax.set_ylabel('Defense')
ax.set_title('Gráfico de dispersión - LOF')
ax.grid(True)
ax.legend(loc='upper left')
plt.show()

# Crea el gráfico de dispersión para KNN
fig, ax = plt.subplots()
ax.scatter(X['Attack'], X['Defense'], marker='o', label='Datos', alpha=0.3, edgecolors='none')
ax.scatter(X[mask_knn]['Attack'], X[mask_knn]['Defense'], marker='x', label='KNN outliers')
ax.set_xlabel('Attack')
ax.set_ylabel('Defense')
ax.set_title('Gráfico de dispersión - KNN')
ax.grid(True)
ax.legend(loc='upper left')
plt.show()

# 2.3. Clustering (35 ptos)

pokemon_no_norm = pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].to_numpy()
pokemon_norm = norm_pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']].to_numpy()

# Finding the optimum number of clusters for k-means classification
wcss = []     
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(pokemon_norm)
    wcss.append(kmeans.inertia_)
# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Sum of squared distances of samples to their closest cluster center
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(pokemon_no_norm)
plt.scatter(pokemon_no_norm[y_kmeans == 0, 0], pokemon_no_norm[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Categoria 1')
plt.scatter(pokemon_no_norm[y_kmeans == 1, 0], pokemon_no_norm[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Categoria 2')
plt.scatter(pokemon_no_norm[y_kmeans == 2, 0], pokemon_no_norm[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Categoria 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroides')
plt.title('K-Means con todo el dataset')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(pokemon_norm)
plt.scatter(pokemon_norm[y_kmeans == 0, 0], pokemon_norm[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Categoria 1')
plt.scatter(pokemon_norm[y_kmeans == 1, 0], pokemon_norm[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Categoria 2')
plt.scatter(pokemon_norm[y_kmeans == 2, 0], pokemon_norm[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Categoria 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroides')
plt.title('K-Means con dataset normalizado')
plt.legend()
plt.show()

dbscan = DBSCAN(eps=0.5, metric='canberra', min_samples=5)
dbscan.fit(pokemon_no_norm)
pca = PCA(n_components=2).fit(pokemon_no_norm)
pca_2d = pca.transform(pokemon_no_norm)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0: 
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise with dataset')
plt.show()

dbscan = DBSCAN(eps=0.5, metric='canberra', min_samples=5)
dbscan.fit(pokemon_norm)
pca = PCA(n_components=2).fit(pokemon_norm)
pca_2d = pca.transform(pokemon_norm)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0: 
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2','Noise'])
plt.title('DBSCAN finds 2 clusters and noise with normalized daataset')
plt.show()