"""
_summary: Este script sirve para crear los modelos que se usaran sobre el dataset con 
ingenieria de caracteristicas
"""

import pandas as pd

# Importar librerias para los modelos
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

proyecto_data_dos = pd.read_csv('../data/processed/features_for_model.csv')
proyecto_data_dos.head()

# ### 5. Construcción de los modelos
X = proyecto_data_dos.drop('NObeyesdad', axis=1)
y = proyecto_data_dos['NObeyesdad']

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=2025, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# #### 5.1. Naive Bayes
resultados_nb =[]

model1_nb = MultinomialNB(alpha=0.5, fit_prior=True)
model2_nb = MultinomialNB(alpha=1.0, fit_prior=True)
model3_nb = MultinomialNB(alpha=1.5, fit_prior=False)

models_nb = [model1_nb, model2_nb, model3_nb]

for i, model in enumerate(models_nb):
    model.fit(x_train_scaled, y_train)
    nb_predicts = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, nb_predicts)
    resultados_nb.append({'Modelo': f'model{i+1}', 'Accuracy': accuracy})


df_resultados = pd.DataFrame(resultados_nb).sort_values(by='Accuracy',
                                                        ascending= False, ignore_index= True)

# #### 5.2. LDA - Análisis de Discriminante Lineal
resultados_lda=[]

model1_lda = LinearDiscriminantAnalysis(solver='svd')
model2_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
model3_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)

models_lda = [model1_lda, model2_lda, model3_lda]

for i, model in enumerate(models_lda):
    model.fit(x_train_scaled, y_train)
    lda_predicts = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, lda_predicts)
    resultados_lda.append({'Modelo': f'model{i+1}', 'Accuracy': accuracy})


df_resultados = pd.DataFrame(resultados_lda).sort_values(by='Accuracy',
                                                         ascending= False, ignore_index= True)

# #### 5.3. Regresión Logística
resultados_lg=[]

model1_lg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=1000, multi_class='multinomial')
model2_lg = LogisticRegression(penalty='l1', C=0.5, solver='saga',
                               max_iter=1000, multi_class='multinomial')
model3_lg = LogisticRegression(penalty='elasticnet', C=1.0, solver='saga',
                               l1_ratio=0.5, max_iter=1000, multi_class='multinomial')

models_lg = [model1_lg, model2_lg, model3_lg]

for i, model in enumerate(models_lg):
    model.fit(x_train_scaled, y_train)
    predicts_lg = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, predicts_lg)
    resultados_lg.append({'Modelo': f'model{i+1}', 'Accuracy': accuracy})


df_resultados = pd.DataFrame(resultados_lg).sort_values(by='Accuracy',
                                                        ascending= False, ignore_index= True)

# #### 5.4. SVM
resultados_svm=[]

model1_SVM = SVC(C=1.0, kernel='linear')
model2_SVM = SVC(C=1.0, kernel='poly', degree=3, gamma='scale')
model3_SVM = SVC(C=1.0, kernel='rbf', gamma='scale')

models_SVM = [model1_SVM, model2_SVM, model3_SVM]

for i, model in enumerate(models_SVM):
    model.fit(x_train_scaled, y_train)
    predicts_SVM = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, predicts_SVM)
    resultados_svm.append({'Modelo': f'model{i+1}', 'Accuracy': accuracy})


df_resultados = pd.DataFrame(resultados_svm).sort_values(by='Accuracy',
                                                         ascending= False, ignore_index= True)

# #### 5.5. Random Forest
resultados_rf=[]

model1_rf = RandomForestClassifier(n_estimators=50,
                                   criterion='gini', max_depth=10)
model2_rf = RandomForestClassifier(n_estimators=100,
                                   criterion='entropy', max_depth=15, min_samples_split=10)
model3_rf = RandomForestClassifier(n_estimators=150,
                                   criterion='gini', max_depth=None, min_samples_leaf=5)

models_rf = [model1_rf, model2_rf, model3_rf]

for i, model in enumerate(models_rf):
    model.fit(x_train_scaled, y_train)
    predicts_rf = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, predicts_rf)
    resultados_rf.append({'Modelo': f'model{i+1}', 'Accuracy': accuracy})

df_resultados = pd.DataFrame(resultados_rf).sort_values(by='Accuracy',
                                                        ascending= False, ignore_index= True)
