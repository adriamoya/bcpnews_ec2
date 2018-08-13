# -*- coding: utf-8 -*-
import boto3
import pickle
import xgboost
from io import BytesIO

# Set up connection to s3
s3 = boto3.resource('s3')

# NIVEL 1: CNN - LSTM - BiLSTM - CNNLSTM
# ======================================

# number of models in the first level
models = ['CNN', 'LSTM', 'BiLSTM', 'CNNLSTM']
num_models = len(models)

# number of text inputs used for modeling in the first level
text_vars = ['summary', 'text', 'title'] #, 'keywords'
num_text_vars = len(text_vars)

# Una primera carga para poder tener las dimensiones del dataset
X, Y, X_subm = loading_data(train_path, news_path, 'title')

# Number of folds
n_folds = 4
# Empty array to store out-of-fold predictions (single column)
S_train_A_scratch = np.zeros((X.shape[0], num_models * num_text_vars))
# Empty array to store temporary test set predictions made in each fold
S_test_temp = np.zeros((X_subm.shape[0], n_folds))
# means of the predictions of all the folds for the test set
S_test_A_scratch = np.zeros((X_subm.shape[0], num_models * num_text_vars))
# Empty list to store scores from each fold
scores = []
# Split initialization
kf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=0)


# Importing xgboost model (random search)
model_path_xgb = 'models/2nStageXGB.dat'

with BytesIO() as data:
    s3.Bucket("bluecaparticles").download_fileobj("models/2nStageXGB.dat", data)
    data.seek(0)    # move back to the beginning after writing
    random_search = pickle.load(data)
