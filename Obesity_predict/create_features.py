"""
_summary: Este notebook sirve para aplicar ingenieria de caracteristicas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

proyecto_data = pd.read_csv('../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv', delimiter=';')
proyecto_data.head()

# 2. Ingeniería de características:

# a. Imputación de variables con data faltante:
# Numérica

#Determinacion de nulos:
col_con_na = []
for col in proyecto_data.columns:
    porcentaje_faltante = proyecto_data[col].isnull().mean()
    if(porcentaje_faltante > 0):
        col_con_na.append(col)

col_con_na

# * Analisis CCA

#Extraer columnas numericas
cols_for_cca = [col for col in col_con_na if proyecto_data[col].dtype in ['float64']]
cols_for_cca

#Imputacion por medio del CCA.
data_cca = proyecto_data[cols_for_cca].dropna()

#Resultado de imputacion por CCA.

for col in cols_for_cca:
    fig = plt.figure(figsize=(4, 3))
    proyecto_data[col].plot.density(color='red', legend='Original')
    data_cca[col].plot.density(color='blue', legend='CCA')
    plt.show()

# Analisis Media

#Calculo para visualizar la media de cada columna numerica
medias = proyecto_data[cols_for_cca].mean()

#Crear copia del dataset para imputacion de media
proyecto_data_media = proyecto_data.copy()

#Imputacion de la media en las columnas numericas
for col in cols_for_cca:
    media_columna = proyecto_data_media[col].mean()
    proyecto_data_media[col] = proyecto_data_media[col].fillna(media_columna)

#Resultados de la imputación
for col in cols_for_cca:
    fig = plt.figure(figsize=(4, 3))
    proyecto_data[col].plot.density(color='red', legend='Original')
    proyecto_data_media[col].plot.density(color='blue', legend='Media')
    plt.show()

# Analisis Mediana

#Calculo de la mediana por cada columna numerica con datos nulos
medianas = proyecto_data[cols_for_cca].median()

proyecto_data_mediana = proyecto_data.copy()

#Imputacion de la media en las columnas numericas
for col in cols_for_cca:
    mediana_columna = proyecto_data_mediana[col].median()
    proyecto_data_mediana[col] = proyecto_data_mediana[col].fillna(mediana_columna)

for col in cols_for_cca:
    fig = plt.figure(figsize=(4, 3))
    proyecto_data[col].plot.density(color='red', legend='Original')
    proyecto_data_mediana[col].plot.density(color='blue', legend='Media')
    plt.show()

# a. Imputación de variables con data faltante:
# Categoricas

#Extraer variables categoricas con nulos.
categoricas = [col for col in col_con_na if proyecto_data_media[col].dtype == 'object']

#Analisis por medio de grafica
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(111)
proyecto_data_media[proyecto_data_media['Gender'] ==
                    'Male' ]['Weight'].plot(kind='kde', ax=ax, label='Male')
proyecto_data_media[proyecto_data_media['Gender'] ==
                    'Female' ]['Weight'].plot(kind='kde', ax=ax, label='Female')
proyecto_data_media[proyecto_data_media['Gender'].isnull()]['Weight'].plot(kind='kde',
                                                                           ax=ax, label='NAN')
plt.legend()
plt.show()

#Imputacion de la variable Gender con una nueva categoria 'Missing'.
proyecto_data_media['Gender'].fillna('Missing', inplace=True)


#Analisis por medio de grafica
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(111)
proyecto_data_media[proyecto_data_media['FAVC'] ==
                    'yes' ]['Weight'].plot(kind='kde', ax=ax, label='yes')
proyecto_data_media[proyecto_data_media['FAVC'] ==
                    'no' ]['Weight'].plot(kind='kde', ax=ax, label='no')
proyecto_data_media[proyecto_data_media['FAVC'].isnull()]['Weight'].plot(kind='kde',
                                                                         ax=ax, label='NAN')
plt.legend()
plt.show()

#Imputacion de la variable FAVC con 'yes'.
proyecto_data_media['FAVC'].fillna('yes', inplace=True)

fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
proyecto_data_media[proyecto_data_media['MTRANS'] ==
                    'Public_Transportation' ]['Weight'].plot(kind='kde', ax=ax, label='Public_Transportation')
proyecto_data_media[proyecto_data_media['MTRANS'] ==
                    'Automobile' ]['Weight'].plot(kind='kde', ax=ax, label='Automobile')
proyecto_data_media[proyecto_data_media['MTRANS'] ==
                    'Walking' ]['Weight'].plot(kind='kde', ax=ax, label='Walking')
proyecto_data_media[proyecto_data_media['MTRANS'] ==
                    'Motorbike' ]['Weight'].plot(kind='kde', ax=ax, label='Motorbike')
proyecto_data_media[proyecto_data_media['MTRANS'] ==
                    'Bike' ]['Weight'].plot(kind='kde', ax=ax, label='Bike')
plt.legend()
plt.show()

#Imputacion de la variable MTRANS con una nueva categoria 'Missing'.
proyecto_data_media['MTRANS'].fillna('Missing', inplace=True)

# b. Codificación de variables categóricas

#Determinar las columnas categóricas
cols_for_cod = [col for col in proyecto_data_media.columns
                if proyecto_data_media[col].dtype in ['object']]

#Leyenda de categorias antes de codificar sobre las variables categoricas
for col in cols_for_cod:
    leyenda = proyecto_data_media[col].value_counts()
    print(leyenda)

#Función para determinar la frecuencia de la variable categórica
def frequency_enconding_procedure(data_serie):
    freq_dict = data_serie.value_counts(ascending=False).to_dict()
    return data_serie.map(freq_dict)

#bucle para codificar las variables categóricas y reemplazar los datos en el dataset
for col in cols_for_cod:
    proyecto_data_media[col] = frequency_enconding_procedure(proyecto_data_media[col])

#Resultado de la codificacion por medio de la frecuencia:
proyecto_data_media.head()

# c. Tratamiento de Outliers

#Función para identificar las variables continuas, discretas y categoricas
def get_variables_scale(dataset):
    categoricas = [col for col in dataset.columns if dataset[col].dtype == 'object']
    continuas = [col for col in dataset.columns if dataset[col].dtype in ['float64','int64']
                and len(dataset[col].unique()) > 30]
    discretas = [col for col in dataset.columns if dataset[col].dtype in ['float64','int64']
                and len(dataset[col].unique()) <= 30]
    return categoricas, continuas, discretas

cat, cont, disct = get_variables_scale(proyecto_data_media)

#Visualizacion de las columnas antes del tratamiento de outliers
def plot_outliers_analysis(dataset, col):
    plt.figure(figsize=(20,6))

    print(col)
    plt.subplot(131)
    sns.histplot(dataset[col], bins=30)
    plt.title("Densidad - Histograma")

    plt.subplot(132)
    stats.probplot(dataset[col], dist="norm", plot=plt)
    plt.title("QQ-Plot")

    plt.subplot(133)
    sns.boxplot(y=dataset[col])
    plt.title("Boxplot")

    plt.show()

for col in cont:
    plot_outliers_analysis(proyecto_data_media, col)

#Tratamiento de Outliers con el metodo Capping
proyecto_data_cap = proyecto_data_media.copy()

#Obtener limite superior e inferior
def get_outliers_limits(dataset,col):
    iqr = proyecto_data_cap[col].quantile(0.75) - proyecto_data_cap[col].quantile(0.25)
    li = proyecto_data_cap[col].quantile(0.25) - (1.5*iqr)
    ls = proyecto_data_cap[col].quantile(0.75) + (1.5*iqr)
    return LI, LS

for col in cont:
    li, ls = get_outliers_limits(proyecto_data_cap, col)

    proyecto_data_cap[col] = np.where(proyecto_data_cap[col] > ls, ls,
         np.where(proyecto_data_cap[col] < li, li,
         proyecto_data_cap[col]))
   
for col in cont:
    plot_outliers_analysis(proyecto_data_cap, col)

# d. Transformación de variables numéricas

#Funcion para visualizar la distribucion de las variables categoricas
def plot_density_qq(df, variable):
    plt.figure(figsize = (8, 3))
    plt. subplot(121)
    df[variable].hist(bins = 30)
    plt.title(variable)

    plt.subplot(122)
    stats.probplot(df[variable], dist='norm', plot=plt)
    plt.show()

for col in cont:
    plot_density_qq(proyecto_data_cap, col)

# - Transformacion Logaritmica

proyecto_data_log = proyecto_data_cap.copy()

for col in cont:
    if col in proyecto_data_log.columns:
        if (proyecto_data_log[col] <= 0).any():
            print(f"Advertencia: La columna '{col}' contiene valores <= 0")
            continue
        proyecto_data_log[col] = np.log(proyecto_data_log[col])
        plot_density_qq(proyecto_data_log, col)
        plt.show()

# - Transformacion Polinomial

proyecto_data_poli = proyecto_data_cap.copy()

for col in cont:
    if col in proyecto_data_poli.columns:
        if (proyecto_data_poli[col] == 0).any():
            print(f"Advertencia: La columna '{col}' contiene valores igual a 0")
            continue
        proyecto_data_poli[col] = (proyecto_data_poli[col]) ** 2
        plot_density_qq(proyecto_data_poli, col)
        plt.show()

# - Transformacion Exponencial
proyecto_data_expo = proyecto_data_cap.copy()

for col in cont:
    if col in proyecto_data_expo.columns:
        if (proyecto_data_expo[col] == 0).any():
            print(f"Advertencia: La columna '{col}' contiene valores igual a 0")
            continue
        proyecto_data_expo[col] = 1 / proyecto_data_expo[col]
        plot_density_qq(proyecto_data_expo, col)
        plt.show()

# - Transformacion Box - Cox

proyecto_data_box = proyecto_data_cap.copy()

for col in cont:
    if col in proyecto_data_box.columns:
        if (proyecto_data_box[col] == 0).any():
            print(f"Advertencia: La columna '{col}' contiene valores igual a 0")
            continue
        proyecto_data_box[col], lmbd= stats.boxcox(proyecto_data_box[col])
        lmbd = str(round(lmbd,4))
        print(lmbd)
        plot_density_qq(proyecto_data_box, col)
        plt.show()

proyecto_data_log.to_csv("../data/processed/features_for_model.csv", index=False)
