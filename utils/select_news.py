
# -*- coding: utf-8 -*-
import os
import boto3
import pickle
import xgboost
import numpy as np
import pandas as pd
import random as rn
from io import BytesIO
# import tensorflow as tf
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold

from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

def prepare_text_data(train_input, test_input, max_words, max_len):
    print('--> Tokenizing and padding data...')
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(train_input)
    sequences_train = tok.texts_to_sequences(train_input)
    sequences_test = tok.texts_to_sequences(test_input)
    print('--> Pad sequences (samples x time)')
    test_input_f = sequence.pad_sequences(sequences_test, maxlen=max_len)
    return test_input_f

def models_fit_predict(crawl_date, df_subm):

    S3_BUCKET = "bluecaparticles"
    path_train = "data/train.csv"
    path_model_xgb = "models/2nStageXGB.dat"

    client = boto3.client('s3') #low-level functional API
    s3 = boto3.resource('s3') #high-level object-oriented API

    # Loading data
    print('\n--> Loading train dataset...')
    obj =  client.get_object(Bucket=S3_BUCKET, Key=path_train)
    df = pd.read_csv(obj['Body'], encoding = 'utf8')

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

#           # Loop del stacking
            for fold_counter in range(N_FOLDS):

                # Returns a compiled model
                # identical to the previous one
                path_model = "models/%s%s%s.h5" % (modeles, feature, fold_counter)
                # with BytesIO() as data:
                # s3.Bucket(S3_BUCKET).download_fileobj(path_model, data)
                # data.seek(0)    # move back to the beginning after writing
                s3.Bucket(S3_BUCKET).download_file(path_model, 'tmp_model.h5')
                model = load_model('tmp_model.h5')
                os.remove('tmp_model.h5')

                # Predict test set
                # Guardamos una predccion de todo el test por fold y luego haremos la media
                S_test_temp[:, fold_counter] = model.predict(X_test, verbose=0).ravel()

                print('fold %d' % fold_counter)

            # Metodo 2. Stacking haciendo media de las predicciones por fold de la parte test
            # -------------------------------------------------------------------------------------------
            # Incorporamos todas las predicciones por fold en un resumen de predicciones por modelo
            # Compute mean of temporary test set predictions to get final test set prediction

            print('--> Averaging scores...')
            S_test_A_scratch[:, (k * num_models) + i] = np.mean(S_test_temp, axis=1).reshape(-1, 1).ravel()


    # NIVEL 2: STACKING
    # =================

    # 2n level training with xgboost A parameter grid for XGBoost
    # --------------------------------------------------------------------------

    print("\n-->Training level 2: xgboost...")

    # Loading 2n Stage XGBoost (random search)
    with BytesIO() as data:
        s3.Bucket(S3_BUCKET).download_fileobj(path_model_xgb, data)
        data.seek(0)    # move back to the beginning after writing
        random_search = pickle.load(data)

    y_subm = random_search.predict_proba(S_test_A_scratch)

    subm = pd.concat([df_subm, pd.DataFrame(y_subm[:,1])], axis=1)
    subm.rename(columns={0:'score'}, inplace=True)

    # Saving to s3
    print('\n--> Saving to s3...')
    csv_buffer = StringIO()
    subm.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET_NAME, 'output/%s_articles_score.csv' % crawl_date).put(Body=csv_buffer.getvalue())



# # here it should return the selection of news
#
# def select_news(data, out_file, threshold):
#     print('Selecting news with probability > {}...'.format(threshold))
#     df = pd.read_csv(data)
#     df['score'] = df['score'].astype(float)
#     BBB = df[df['score'] > threshold].sort_values(by='score', ascending=False)
#     BBB = BBB[BBB.url.str.contains("#Comentarios") == False]
#     BBB = BBB[BBB.url.str.contains("#comentarios") == False]
#     BBB = BBB.drop_duplicates(['text'])
#     BBB.to_csv(out_file, index=False)
#     print('\nThese are the news selected (before checking if already existed in previous emails):\n')
#     print(BBB.groupby('newspaper').size().reset_index().rename(columns={0:'count'}).head(20))
#     print('\nDetail:\n')
#     print(BBB.groupby(['newspaper', 'publish_date']).size().rename(columns={0:'count'}).head(20))
#     print('')
#
#
# def save_into_db(database, selected, final_BBB_file):
#     is_first = True
#     final_BBB = ""
#     connection = create_connection(database)
#     sql_create_bbb_news_table = """ CREATE TABLE IF NOT EXISTS BBBNews (
#                                         id integer PRIMARY KEY,
#                                         authors text,
#                                         keywords text,
#                                         publish_date text,
#                                         summary text,
#                                         text text,
#                                         title text,
#                                         top_image text,
#                                         url text NOT NULL,
#                                         score text,
#                                         newspaper text
#                                     ); """
#     if connection is not None:
#         # create projects table
#         create_table(connection, sql_create_bbb_news_table)
#
#     else:
#         print("Error! cannot create the database connection.")
#
#     with connection:
#         df = pd.read_csv(selected)
#         for index, row in df.iterrows():
#             print(row.url)
#             if (check_if_exists(connection, row.url) == 0):
#                 new = (row.authors, row.keywords, row.publish_date, row.summary, row.text, row.title, row.top_image, row.url, row.score, row.newspaper)
#                 insert_news(connection, new)
#                 if is_first:
#                     s1 = pd.Series([row.authors, row.keywords, row.publish_date, row.summary, row.text, row.title, row.top_image, row.url, row.score, row.newspaper,])
#                     final_BBB = pd.DataFrame([list(s1)], columns=['authors','keywords','publish_date','summary','text','title','top_image','url','score','newspaper'])
#                     is_first = False
#                 else:
#                     s1 = pd.Series([row.authors, row.keywords, row.publish_date, row.summary, row.text, row.title, row.top_image, row.url, row.score, row.newspaper])
#                     can = pd.DataFrame([list(s1)], columns=['authors','keywords','publish_date','summary','text','title','top_image','url','score','newspaper'])
#                     final_BBB = final_BBB.append(can)
#
#     connection.close()
#
#     try:
#         final_BBB.to_csv(final_BBB_file, index=False)
#         print('\nThese are the news selected:\n')
#         print(final_BBB.groupby('newspaper').size().reset_index().rename(columns={0:'count'}))
#         print('\nDetail:\n')
#         print(final_BBB.groupby(['publish_date', 'newspaper']).size().reset_index().rename(columns={0:'count'}))
#         print('')
#     except:
#         pass
#
#
# # print('Sending the email to everybody...')
# # print('------------------------------------------------------')
# # email = EmailSender('amoya@bluecap.com', 'tarra1991')
# # email.send_mail(['amoya@bluecap.com'], '%s Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_selection_xgb), 'html')
# # email.send_mail(['egilabert@bluecap.com'], '%s XGBOOST Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_selection_xgb), 'html')
