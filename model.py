# -*- coding: utf-8 -*-
import boto3
import pickle
import xgboost
import numpy as np
import pandas as pd
import random as rn
from io import BytesIO
# import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# def models_fit_predict():

S3_BUCKET = "bluecaparticles"
path_train = "data/train.csv"
path_model_xgb = "models/2nStageXGB.dat"

client = boto3.client('s3') #low-level functional API
# resource = boto3.resource('s3') #high-level object-oriented API
# my_bucket = resource.Bucket('my-bucket') #subsitute this for your s3 bucket name.


# Loading data
obj =  client.get_object(Bucket=S3_BUCKET, Key=path_train)
df = pd.read_csv(obj['Body'], encoding = 'utf8')

# load downloaded articles
articles_downloaded = []
path_articles = [x['Key'] for x in client.list_objects(Bucket=S3_BUCKET)['Contents'] if 'articles' in x['Key']]
for idx, path_newspaper_articles in enumerate(path_articles):
    newspaper_articles = client.get_object(Bucket=S3_BUCKET, Key=path_newspaper_articles)
    # if idx == 1:
    print(idx, path_newspaper_articles)
    df_temp = pd.read_csv(newspaper_articles['Body'], encoding='utf8')
    print(df_temp.shape)
    articles_downloaded.append(df_temp)
    del df_temp
    df_subm = pd.concat(articles_downloaded)

    print('Articles downloaded...')
    print(df_subm.shape)

def prepare_text_data(train_input, test_input, max_words, max_len):
    print('Tokenizing and padding data...')
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(train_input)
    sequences_train = tok.texts_to_sequences(train_input)
    sequences_test = tok.texts_to_sequences(test_input)
    print('Pad sequences (samples x time)')
    test_input_f = sequence.pad_sequences(sequences_test, maxlen=max_len)
    return test_input_f

# NIVEL 1: CNN - LSTM - BiLSTM - CNNLSTM
# ======================================

# number of models in the first level
models = ["CNN", "LSTM", "BiLSTM", "CNNLSTM"]
num_models = len(models)

# number of text inputs used for modeling in the first level
text_vars = ["summary", "text", "title"] #, "keywords"
num_text_vars = len(text_vars)

# for var in text_vars:
#     df_subm[var] = df_subm[var].astype(str)

N_FOLDS = 4

# Empty array to store temporary test set predictions made in each fold
S_test_temp = np.zeros((df_subm.shape[0], N_FOLDS))
# means of the predictions of all the folds for the test set
S_test_A_scratch = np.zeros((df_subm.shape[0], num_models * num_text_vars))

# Split initialization
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False, random_state=0)

# Loop accross features
for k, feature in enumerate(text_vars):
    # Cargamos la feature dinamicamente
    print(feature)
    print(df.columns)
    print(df_subm.columns)
    X = df[feature]
    X_subm = df_subm[feature]
    # Loop accross models
    for i, modeles in enumerate(models):
        # Loop across folds
        print('')
        print('-'*80)
        print('Model {} of {}. Neural network architechture {} using "{}" as feature '.format((k*num_models)+i+1, num_models * num_text_vars, modeles, feature))
        print('-'*80)

        # Manera cutre de definir los parametros de cada uno de los modelos (no me juzguéis por eso, era un WIP...)
        if i == 0:
            max_features = 5000
            maxlen = 400
            X_test = prepare_text_data(X, X_subm, max_features, maxlen)
        elif i == 1:
            max_features = 20000
            maxlen = 80
            X_test = prepare_text_data(X, X_subm, max_features, maxlen)
        elif i == 2:
            max_features = 20000
            maxlen = 100
            X_test = prepare_text_data(X, X_subm, max_features, maxlen)
        elif i == 3:
            max_features = 20000
            maxlen = 100
            X_test = prepare_text_data(X, X_subm, max_features, maxlen)

        # Loop del stacking
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, Y)):

            # # returns a compiled model
            # # identical to the previous one
            # model = load_model(os.path.join(dirname, 'models')+'/'+str(modeles)+str(feature)+str(fold_counter)+'.h5')
            #
            # # Predict test set
            # # Guardamos una predccion de todo el test por fold y luego haremos la media
            # S_test_temp[:, fold_counter] = model.predict(X_test, verbose=0).ravel()

            print('fold %d' % fold_counter)

        # # Metodo 2. Stacking haciendo media de las predicciones por fold de la parte test
        # # -------------------------------------------------------------------------------------------
        # # Incorporamos todas las predicciones por fold en un resumen de predicciones por modelo
        # # Compute mean of temporary test set predictions to get final test set prediction
        #
        # S_test_A_scratch[:, (k * num_models) + i] = np.mean(S_test_temp, axis=1).reshape(-1, 1).ravel()
        # # Mean OOF score + std
        # print('\nMEAN:   [%.8f] + [%.8f]' % (np.mean(scores), np.std(scores)))
