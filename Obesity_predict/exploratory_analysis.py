"""Exploración de datos

    Returns:
        _type_: analisis y exploracion
"""
# %% [markdown]
# ### Proyecto - Análisis exploratorio

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
datos =pd.read_csv("../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv",
                               delimiter=';')
datos.head()

# %% [markdown]
# #### Columnas por tipo de dato

def get_variables_scale(datos):
    categoricas = [col for col in datos.columns if datos[col].dtype == 'object']
    continuas = ([col for col in datos.columns if datos[col].dtype in ['float64','int64'] and 
                  len(datos[col].unique()) > 30])
    discretas = ([col for col in datos.columns if datos[col].dtype in ['float64','int64'] and 
                  len(datos[col].unique()) <= 30])
    return categoricas, continuas, discretas

# %%
cat, cont, disct = get_variables_scale(datos)


# %% [markdown]
# #### Variables categóricas

# %%
col_categorica = cat

for col in col_categorica:
    unique_values_counts = datos[col].value_counts().reset_index()
    unique_values_counts.columns = [f'Categoría {col}', 'Count']
    print(f'Valores en variable: {col}')
    print(unique_values_counts)
    print('\n')



# %%
for col in col_categorica:
    #Frecuencia de cada categoría en la columna
    counts = datos[col].value_counts()
    plt.figure(figsize=(7, 3))
    counts.plot(kind='bar', color='skyblue')    
    
    # Título y etiquetas de los ejes
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=50)
    plt.show()

# %% [markdown]
# #### Variables numericas

# %%
col_numeric = cont

datos[col_numeric].describe()

# %%
for col in col_numeric:
    # Crear una figura nueva para cada gráfica
    plt.figure(figsize=(7, 3))

    # Crear el histograma y gráfico de densidad
    sns.histplot(datos[col], kde=True, bins=30)
    # Título y etiquetas de los ejes
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    # Mostrar la gráfica
    plt.show()
