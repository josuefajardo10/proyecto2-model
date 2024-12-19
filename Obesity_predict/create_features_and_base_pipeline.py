""" El siguiente script sirve para configurar el pipeline de ingenieria de caracteristicas
"""

import configparser
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import LogTransformer
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("../data/raw/ObesityDataSet_raw_and_data_sinthetic.csv", delimiter=';')

config = configparser.ConfigParser()
config.read('../pipeline.cfg')

frecuency_target = dataset['NObeyesdad'].value_counts(ascending=False).to_dict()
dataset['NObeyesdad'] = dataset['NObeyesdad'].map(frecuency_target)
dataset.head()

drop_vars = list(config.get('GENERAL', 'VARS_TO_DROP').split(', '))

x_features = dataset.drop(labels=drop_vars, axis=1)
y_target = dataset['NObeyesdad']
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target,
                                                    test_size=0.3, shuffle=True, random_state=2025)

obesity_predict_model = Pipeline([
    # imputación de variables continuas.
    ('continues_var_imputation', MeanMedianImputer(
        imputation_method='mean',variables=config.get('CONTINUES',
                                                      'CONTINUE_VARS_TO_IMPUTATION').split(', ') )),

    # imputación de variables categóricas
    ('categorical_var_imputation', CategoricalImputer(
        imputation_method='frequent', variables=config.get('CATEGORICAL',
                                                           'CATEGORICAL_VARS_TO_IMPUTATION').split(', ') )),

    # codificación de variables categoricas
    ('categorical_encode_frequency', CountFrequencyEncoder(
        encoding_method='count', variables=config.get('CATEGORICAL',
                                                      'FREQENC_VARS_TO_ENCODE').split(', ') )),

    # tratamiento de outliers
    ('continues_var_outliers', Winsorizer(capping_method='iqr',
                                          variables=config.get('CONTINUES',
                                                               'CONTINUE_VARS_TO_OUTLIERS').split(', ') )),

    # transformacion de variables
    ('continues_var_transform', LogTransformer(
        variables=config.get('CONTINUES', 'CONTINUE_VARS_TO_TRANSFORM').split(', ') )),

    # feature scaling
    ('feature_scaling', MinMaxScaler())
])

# configuramos pipeline
obesity_predict_model.fit(x_train)

x_features_processed = obesity_predict_model.transform(x_train)
df_features_process = pd.DataFrame(x_features_processed, columns=x_train.columns)
df_features_process['NObeyesdad'] = y_train.reset_index()['NObeyesdad']

# guardamos los datos para entrenar los modelos.
df_features_process.to_csv('../data/processed/features_for_model.csv', index=False)

x_test['NObeyesdad'] = y_test
x_test.to_csv('../data/processed/test_dataset.csv', index=False)

with open('../artifacts/pipeline.pkl', 'wb') as f:
    pickle.dump(obesity_predict_model, f)
