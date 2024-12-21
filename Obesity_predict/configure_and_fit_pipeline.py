# %%
import pandas as pd
import pickle
# importaciones
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# %%
data_train = pd.read_csv('../data/processed/features_for_model.csv')
data_test = pd.read_csv('../data/processed/test_dataset.csv')
#data_train.head()

# %%
x_features = data_train.drop(['NObeyesdad'], axis=1)
y_target = data_train['NObeyesdad']

x_features_test = data_test.drop(['NObeyesdad'], axis=1)
y_target_test = data_test['NObeyesdad']
# %% [markdown]
# ### Leemos el Pipeline pre-configurado

# %%
with open('../artifacts/pipeline.pkl', 'rb') as  f:
    obesity_predict_model_pipeline = pickle.load(f)

#titanic_survived_model_pipeline

# %%
x_features_test_arr = obesity_predict_model_pipeline.transform(x_features_test)
df_features_test = pd.DataFrame(x_features_test_arr, columns=x_features_test.columns)
df_features_test.head()

# %% [markdown]
# Entrenamiento de Modelos

# %%
import mlflow

# %%
# configuración de servidor
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Obesity Predict Model")
# %%
model_accuracies = {}
# %%
with mlflow.start_run(run_name="Random Forest Model 1"):
    rf_model = RandomForestClassifier(n_estimators=200, criterion='gini').fit(x_features, y_target)
    y_preds = rf_model.predict(df_features_test)
    acc_rf = accuracy_score(y_target_test, y_preds)
    params_vals = dict(n_estimators=200, criterion='gini')
    model_accuracies["Random Forest Model 1"] = acc_rf
    mlflow.log_params(params_vals)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.sklearn.log_model(rf_model, "random_forest_model_1")

# %%
with mlflow.start_run(run_name="Random Forest Model 2"):
    rf_model_2 = (RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=15, min_samples_split=10).fit(x_features, y_target))
    y_preds_2 = rf_model_2.predict(df_features_test)
    acc_rf_2 = accuracy_score(y_target_test, y_preds_2)
    print(acc_rf_2)
    params_vals_2 = dict(n_estimators=100, criterion='entropy', max_depth=15, min_samples_split=10)
    model_accuracies["Random Forest Model 2"] = acc_rf_2
    mlflow.log_params(params_vals_2)
    mlflow.log_metric("accuracy score Random Forest 2:", acc_rf_2)
    mlflow.sklearn.log_model(rf_model_2, "Random Forest Classifier 2")

# %%
with mlflow.start_run(run_name="Random Forest Model 3"):
    rf_model_3 = (RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_leaf=5).fit(x_features, y_target))
    y_preds_3 = rf_model_3.predict(df_features_test)
    acc_rf_3 = accuracy_score(y_target_test, y_preds_3)
    print(acc_rf_3)
    params_vals_3 = dict(n_estimators=150, criterion='gini', max_depth=None, min_samples_leaf=5)
    model_accuracies["Random Forest Model 3"] = acc_rf_3
    mlflow.log_params(params_vals_3)
    mlflow.log_metric("accuracy score Random Forest 3:", acc_rf_3)
    mlflow.sklearn.log_model(rf_model_3, "Random Forest Classifier 3")

# %%
with mlflow.start_run(run_name="SVM 1"):
    svm_model = SVC(C=1.0, kernel='linear').fit(x_features, y_target)
    y_preds_4 = svm_model.predict(df_features_test)
    acc_svm = accuracy_score(y_target_test, y_preds_4)
    print(acc_svm)
    params_vals_4 = dict(C=1.0, kernel='linear')
    model_accuracies["SVM 1"] = acc_svm
    mlflow.log_params(params_vals_4)
    mlflow.log_metric("accuracy score SVM 1:", acc_svm)
    mlflow.sklearn.log_model(svm_model, "SVM 1")

# %%
with mlflow.start_run(run_name="SVM 2"):
    svm_model_2 = SVC(C=1.0, kernel='poly', degree=3, gamma='scale').fit(x_features, y_target)
    y_preds_5 = svm_model_2.predict(df_features_test)
    acc_svm_2 = accuracy_score(y_target_test, y_preds_5)
    print(acc_svm_2)
    params_vals_5 = dict(C=1.0, kernel='poly', degree=3, gamma='scale')
    model_accuracies["SVM 2"] = acc_svm_2
    mlflow.log_params(params_vals_5)
    mlflow.log_metric("accuracy score SVM 2:", acc_svm_2)
    mlflow.sklearn.log_model(svm_model_2, "SVM 2")

# %%
with mlflow.start_run(run_name="SVM 3"):
    svm_model_3 = SVC(C=1.0, kernel='rbf', gamma='scale').fit(x_features, y_target)
    y_preds_6 = svm_model_3.predict(df_features_test)
    acc_svm_3 = accuracy_score(y_target_test, y_preds_6)
    print(acc_svm_3)
    params_vals_6 = dict(C=1.0, kernel='rbf', gamma='scale')
    model_accuracies["SVM 3"] = acc_svm_3
    mlflow.log_params(params_vals_6)
    mlflow.log_metric("accuracy score SVM 3:", acc_svm_3)
    mlflow.sklearn.log_model(svm_model_3, "SVM 3")

# %%
with mlflow.start_run(run_name="Regresion Logistica 1"):
    lr_model = LogisticRegression().fit(x_features, y_target)
    y_preds_lr = lr_model.predict(df_features_test)
    acc_lr = accuracy_score(y_target_test, y_preds_lr)
    print(acc_lr)
    params_lr = dict(penalty_lr='l2', tol_lr=0.0001, C_lr=1.0)
    model_accuracies["Regresion Logistica 1"] = acc_lr
    mlflow.log_params(params_lr)
    mlflow.log_metric("accuracy score - Regresion Logistica 1:", acc_lr)
    mlflow.sklearn.log_model(lr_model, "Logistic Regression")

# %%
with mlflow.start_run(run_name="Regresion Logistica 2"):
    lr_model_2 = (LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', 
                max_iter=1000, multi_class='multinomial').fit(x_features, y_target))
    y_preds_lr_2 = lr_model_2.predict(df_features_test)
    acc_lr_2 = accuracy_score(y_target_test, y_preds_lr_2)
    print(acc_lr_2)
    params_lr_2 = dict(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    model_accuracies["Regresion Logistica 2"] = acc_lr_2
    mlflow.log_params(params_lr_2)
    mlflow.log_metric("accuracy score - Regresion Logistica 2:", acc_lr_2)
    mlflow.sklearn.log_model(lr_model_2, "Logistic Regression 2")

# %%
with mlflow.start_run(run_name="Regresion Logistica 3"):
    lr_model_3 = (LogisticRegression(penalty='l1', C=0.5, solver='saga', 
                max_iter=1000, multi_class='multinomial').fit(x_features, y_target))
    y_preds_lr_3 = lr_model_3.predict(df_features_test)
    acc_lr_3 = accuracy_score(y_target_test, y_preds_lr_3)
    print(acc_lr_3)
    params_lr_3 = dict(penalty='l1', C=0.5, solver='saga', max_iter=1000, multi_class='multinomial')
    model_accuracies["Regresion Logistica 3"] = acc_lr_3
    mlflow.log_params(params_lr_3)
    mlflow.log_metric("accuracy score - Regresion Logistica 3:", acc_lr_3)
    mlflow.sklearn.log_model(lr_model_3, "Logistic Regression 3")

# %%
with mlflow.start_run(run_name="Naive Bayes 1"):
    nb_model_1 = MultinomialNB(alpha=0.5, fit_prior=True).fit(x_features, y_target)
    y_preds_nb = nb_model_1.predict(df_features_test)
    acc_nb_1 = accuracy_score(y_target_test, y_preds_nb)
    print(acc_nb_1)
    params_vals_nb = dict(alpha=0.5, fit_prior=True)
    model_accuracies["Naive Bayes 1"] = acc_nb_1
    mlflow.log_params(params_vals_nb)
    mlflow.log_metric("accuracy score - Naive Bayes 1:", acc_nb_1)
    mlflow.sklearn.log_model(nb_model_1, "Naive Bayes model 1")

# %%
with mlflow.start_run(run_name="Naive Bayes 2"):
    nb_model_2 = MultinomialNB(alpha=1.0, fit_prior=True).fit(x_features, y_target)
    y_preds_nb2 = nb_model_2.predict(df_features_test)
    acc_nb_2 = accuracy_score(y_target_test, y_preds_nb2)
    print(acc_nb_2)
    params_vals_nb2 = dict(alpha=1.0, fit_prior=True)
    model_accuracies["Naive Bayes 2"] = acc_nb_2
    mlflow.log_params(params_vals_nb2)
    mlflow.log_metric("accuracy score - Naive Bayes 2:", acc_nb_2)
    mlflow.sklearn.log_model(nb_model_2, "Naive Bayes model 2")

# %%
with mlflow.start_run(run_name="Naive Bayes 3"):
    nb_model_3 = MultinomialNB(alpha=1.5, fit_prior=False).fit(x_features, y_target)
    y_preds_nb3 = nb_model_3.predict(df_features_test)
    acc_nb_3 = accuracy_score(y_target_test, y_preds_nb3)
    print(acc_nb_3)
    params_vals_nb3 = dict(alpha=1.5, fit_prior=False)
    model_accuracies["Naive Bayes 3"] = acc_nb_3
    mlflow.log_params(params_vals_nb3)
    mlflow.log_metric("accuracy score - Naive Bayes 3:", acc_nb_3)
    mlflow.sklearn.log_model(nb_model_3, "Naive Bayes model 3")

# %%
with mlflow.start_run(run_name="LDA 1"):
    lda_model_1 = LinearDiscriminantAnalysis(solver='svd').fit(x_features, y_target)
    y_preds_lda = lda_model_1.predict(df_features_test)
    acc_lda_1 = accuracy_score(y_target_test, y_preds_lda)
    print(acc_lda_1)
    params_vals_lda = dict(solver='svd')
    model_accuracies["LDA 1"] = acc_lda_1
    mlflow.log_params(params_vals_lda)
    mlflow.log_metric("accuracy score - LDA 1:", acc_lda_1)
    mlflow.sklearn.log_model(lda_model_1, "LDA model 1")

# %%
with mlflow.start_run(run_name="LDA 2"):
    lda_model_2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(x_features, y_target)
    y_preds_lda2 = lda_model_2.predict(df_features_test)
    acc_lda_2 = accuracy_score(y_target_test, y_preds_lda2)
    print(acc_lda_2)
    params_vals_lda2 = dict(solver='lsqr', shrinkage='auto')
    model_accuracies["LDA 2"] = acc_lda_2
    mlflow.log_params(params_vals_lda2)
    mlflow.log_metric("accuracy score - LDA 2:", acc_lda_2)
    mlflow.sklearn.log_model(lda_model_2, "LDA model 2")

# %%
with mlflow.start_run(run_name="LDA 3"):
    lda_model_3 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5).fit(x_features, y_target)
    y_preds_lda3 = lda_model_3.predict(df_features_test)
    acc_lda_3 = accuracy_score(y_target_test, y_preds_lda3)
    print(acc_lda_3)
    params_vals_lda3 = dict(solver='lsqr', shrinkage=0.5)
    model_accuracies["LDA 3"] = acc_lda_3
    mlflow.log_params(params_vals_lda3)
    mlflow.log_metric("accuracy score - LDA 3:", acc_lda_3)
    mlflow.sklearn.log_model(lda_model_3, "LDA model 3")

# %%
#Identificar el modelo con mejor accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]
print(f"El modelo con mayor accuracy es: {best_model_name} con un accuracy de {best_accuracy:.4f}")

# %%
#Agregando el modelo ganador al pipeline
if best_model_name == "Random Forest Model 1":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_random_forest 1', RandomForestClassifier(n_estimators=200, criterion='gini'))
    )
elif best_model_name == "Random Forest Model 2":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_random_forest 2', RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=15, min_samples_split=10))
    )
elif best_model_name == "Random Forest Model 3":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_random_forest 3', RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_leaf=5))
    )

elif best_model_name == "SVM 1":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_svm 1', SVC(C=1.0, kernel='linear'))
    )
elif best_model_name == "SVM 2":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_svm 2', SVC(C=1.0, kernel='poly', degree=3, gamma='scale'))
    )
elif best_model_name == "SVM 3":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_svm 3', SVC(C=1.0, kernel='rbf', gamma='scale'))
    )

elif best_model_name == "Regresion Logistica 1":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_regresion_logistica 1', LogisticRegression())
    )
elif best_model_name == "Regresion Logistica 2":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_regresion_logistica 2', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial'))
    )
elif best_model_name == "Regresion Logistica 3":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_regresion_logistica 3', LogisticRegression(penalty='l1', C=0.5, solver='saga', max_iter=1000, multi_class='multinomial'))
    )

elif best_model_name == "Naive Bayes 1":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_naive_bayes 1', MultinomialNB(alpha=0.5, fit_prior=True))
    )
elif best_model_name == "Naive Bayes 2":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_naive_bayes 2', MultinomialNB(alpha=1.0, fit_prior=True))
    )
elif best_model_name == "Naive Bayes 3":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_naive_bayes 3', MultinomialNB(alpha=1.5, fit_prior=False))
    )
elif best_model_name == "LDA 1":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_lda 1', LinearDiscriminantAnalysis(solver='svd'))
    )
elif best_model_name == "LDA 2":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_lda 2', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    )
elif best_model_name == "LDA 3":
    obesity_predict_model_pipeline.steps.append(
        ('modelo_lda 3', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5))
    )
# %%
obesity_predict_model_pipeline

# %%
with open('../artifacts/pipeline.pkl', 'wb') as f:
    pickle.dump(obesity_predict_model_pipeline, f)


