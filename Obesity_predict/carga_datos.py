""""
Script para cargar y procesar datos.
Autor: Josue Fajardo
Fecha: 01122024
Descripción: Este script realiza la carga de datos desde un archivo CSV
y los prepara para su análisis posterior.
"""

import pandas as pd

proyecto_data = pd.read_csv("../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv", delimiter=';')
proyecto_data.head()

# %% [markdown]
# ### 1. Selección de Variables:
# Variable a predecir: NObeyesdad, que indica el nivel de obesidad de las personas.
# # Variables predictoras: Entre las variables elegidas estan:
#  * Weight: Peso de las personas.
#  * family_history_with_overweight: Indica si la personas tiene o tuvo familiareas con sobrepeso.
#  * FAVC: Si la persona come alimentos ricos en calorias.
#  * FCVC: Si la persona come vegetales.
#  * NCP: Cuantas comidas principales hace al dia.
#  * SCC: Si la persona controla las calorias que consume diariamente.
#  * FAF: Con que frecuencia la persona hace ejercicio.
#  * CALC: Frecuencia con que bebe alcohol las personas.
