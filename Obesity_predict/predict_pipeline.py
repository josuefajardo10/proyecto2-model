"""
    _summary: Este script sirve para realizar predicciones
"""
import os
import pickle
from datetime import datetime
import pandas as pd

with open('../artifacts/pipeline_final.pkl', 'rb') as f:
    obesity_predict_model = pickle.load(f)

test_dataset = pd.read_csv('../data/processed/test_dataset.csv')
test_dataset_features = test_dataset.drop(['NObeyesdad'], axis=1)
test_dataset_target = test_dataset['NObeyesdad']
obesity_predict_model.fit(test_dataset_features, test_dataset_target)

predict = obesity_predict_model.predict(test_dataset_features)

predictions_dir = '../data/predictions/predict_pipeline'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = f'{predictions_dir}/{timestamp}.csv'

predictions_df = pd.DataFrame(predict, columns=['obesity_prediction'])
predictions_df.to_csv(filename, index=False)
