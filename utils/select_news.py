
import os
import itertools
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
import pickle
from xgboost import XGBClassifier

from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split

from keras.models import load_model, Model, Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from utils.database import check_if_exists, create_connection, create_table, insert_news
from utils.mail_body_generator import create_email_body
from utils.send_email import EmailSender


def models_fit_predict(crawl_date):

    # tf.logging.set_verbosity(ERROR)

    # VARIABLES
    # =========

    # Running date.
    fecha = crawl_date.strftime("%Y%m%d")
    # Path to `YYYYMMDD_articles.json`. Articles downloaded today.
    news_path = './output/%s_articles.json' % fecha
    # Path to training dataset.
    train_path = 'data/train.csv'
    # Path to models.
    model_path_lasso = 'models/2nStageLasso.pkl'
    model_path_xgb = 'models/2nStageXGB.dat'
    # Predicitions on articles downloaded for crawl_date`.
    model_results_lasso = 'data/%s_01_results_lasso.csv' % fecha
    model_results_xgb = 'data/%s_01_results_xgb.csv' % fecha
    # Selection of articles (prediction > 50%).
    news_selection_lasso = 'data/%s_02_selected_bbb_lasso.csv' % fecha
    news_selection_xgb = 'data/%s_02_selected_bbb_xgb.csv' % fecha
    # Final selection of articles after checking if they exists in database.
    final_selection_xgb = 'data/%s_03_final_bbb_xgb.csv' % fecha
    final_selection_lasso = 'data/%s_03_final_bbb_lasso.csv' % fecha
    # Database locationself.
    database = './BBB_database.db'

    dirname = os.path.dirname(os.path.dirname(__file__))

    def loading_data(train_path, test_path, feature):
        print('Loading data with variable feature {}...'.format(feature))
        fullpath_train = os.path.join(dirname, train_path)
        df = pd.read_csv(fullpath_train)
        # fullpath_test = os.path.join(dirname, test_path)
        df_subm = pd.read_json(test_path, lines=True)
        vars = ['summary', 'text', 'title', 'keywords']
        df_subm = df_subm[vars]
        for var in vars:
            df_subm[var] = df_subm[var].astype(str)
        X = df[feature]
        X_subm = df_subm[feature]
        Y = df.flag
        return X, Y, X_subm

    def prepare_text_data(train_input, test_input, max_words, max_len):
        print('Tokenizing and padding data...')
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(train_input)
        sequences_train = tok.texts_to_sequences(train_input)
        sequences_test = tok.texts_to_sequences(test_input)

        print('Pad sequences (samples x time)')
        train_input_f = sequence.pad_sequences(sequences_train, maxlen=max_len)
        test_input_f = sequence.pad_sequences(sequences_test, maxlen=max_len)
        return train_input_f, test_input_f


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

    # Loop accross features
    for k, feature in enumerate(text_vars):
        # Cargamos la feature dinamicamente
        X, Y, X_subm = loading_data(train_path, news_path, feature)
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
                X_train, X_test = prepare_text_data(X, X_subm, max_features, maxlen)
            elif i == 1:
                max_features = 20000
                maxlen = 80
                X_train, X_test = prepare_text_data(X, X_subm, max_features, maxlen)
            elif i == 2:
                max_features = 20000
                maxlen = 100
                X_train, X_test = prepare_text_data(X, X_subm, max_features, maxlen)
            elif i == 3:
                max_features = 20000
                maxlen = 100
                X_train, X_test = prepare_text_data(X, X_subm, max_features, maxlen)

            scores = []

            # Loop del stacking
            for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, Y)):

                # Split data and target
                X_tr = X_train[tr_index]
                y_tr = Y[tr_index]
                X_te = X_train[te_index]
                y_te = Y[te_index]

                # returns a compiled model
                # identical to the previous one
                model = load_model(os.path.join(dirname, 'models')+'/'+str(modeles)+str(feature)+str(fold_counter)+'.h5')

                # Predict out-of-fold part of train set
                # Guardamos las predicciones CV de la parte de train no utilzada en entrenamiento
                S_train_A_scratch[te_index, (k*num_models)+i] = model.predict(X_te, verbose=0).reshape(-1, 1).ravel()

                # Predict test set
                # Guardamos una predccion de todo el test por fold y luego haremos la media
                S_test_temp[:, fold_counter] = model.predict(X_test, verbose=0).ravel()

                # Print score of current fold
                score = roc_auc_score(y_te, S_train_A_scratch[te_index, i])
                scores.append(score)
                print('fold %d: [%.8f]' % (fold_counter, score))

            # Metodo 2. Stacking haciendo media de las predicciones por fold de la parte test
            # -------------------------------------------------------------------------------------------
            # Incorporamos todas las predicciones por fold en un resumen de predicciones por modelo
            # Compute mean of temporary test set predictions to get final test set prediction

            S_test_A_scratch[:, (k * num_models) + i] = np.mean(S_test_temp, axis=1).reshape(-1, 1).ravel()
            # Mean OOF score + std
            print('\nMEAN:   [%.8f] + [%.8f]' % (np.mean(scores), np.std(scores)))


    # NIVEL 2: STACKING
    # =================

    # 2n level training with LASSO
    # --------------------------------------------------------------------------

    print("Training level 2: lasso...")

    clf = LassoCV()
    clf.fit(S_train_A_scratch, Y)

    # joblib.dump(clf, model_path_lasso)
    # clf = joblib.load(model_path_lasso)

    cv_train_auc = roc_auc_score(Y, clf.predict(S_train_A_scratch))
    print('CV train with LASSO AUC: {}'.format(cv_train_auc))

    y_subm = clf.predict(S_test_A_scratch)
    # fullpath_test = os.path.join(dirname, news_path)
    df_news = pd.read_json(news_path, lines=True)
    subm = pd.concat([df_news, pd.DataFrame(y_subm)], axis=1)
    subm.rename(columns={0:'score'}, inplace=True)

    print("Saving results...")
    subm.to_csv(model_results_lasso, index=False)

    # 2n level training with xgboost A parameter grid for XGBoost
    # --------------------------------------------------------------------------

    print("\nTraining level 2: xgboost...")

    """

    params = {
            'min_child_weight': [1, 3, 5],
            'gamma': [0.5, 1, 1.5, 2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.005]
            }

    xgb = XGBClassifier(learning_rate=0.001, n_estimators=10000,
                        objective='binary:logistic', silent=True)
    folds = 4
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    print("Randomized search...")
    random_search = RandomizedSearchCV(xgb,
                                       param_distributions=params,
                                       n_iter=param_comb,
                                       scoring='roc_auc',
                                       n_jobs=-1,
                                       cv=skf.split(S_train_A_scratch,Y),
                                       verbose=1,  # 2
                                       random_state=1001 )

    random_search.fit(S_train_A_scratch, Y)

    pickle.dump(random_search.best_estimator_, open(model_path_xgb, "wb"))
    """

    random_search = pickle.load(open(model_path_xgb, "rb"))

    cv_train_auc = roc_auc_score(Y, random_search.predict_proba(S_train_A_scratch)[:,1])
    print('CV train with XGBoost AUC: {}'.format(cv_train_auc))

    y_subm = random_search.predict_proba(S_test_A_scratch)
    df_news = pd.read_json(news_path, lines=True)
    subm = pd.concat([df_news, pd.DataFrame(y_subm[:,1])], axis=1)
    subm.rename(columns={0:'score'}, inplace=True)

    print("Saving results...")
    subm.to_csv(model_results_xgb, index=False)


# here it should return the selection of news

def select_news(data, out_file, threshold):
    print('Selecting news with probability > {}...'.format(threshold))
    df = pd.read_csv(data)
    df['score'] = df['score'].astype(float)
    BBB = df[df['score'] > threshold].sort_values(by='score', ascending=False)
    BBB = BBB[BBB.url.str.contains("#Comentarios") == False]
    BBB = BBB[BBB.url.str.contains("#comentarios") == False]
    BBB = BBB.drop_duplicates(['text'])
    BBB.to_csv(out_file, index=False)
    print('\nThese are the news selected (before checking if already existed in previous emails):\n')
    print(BBB.groupby('newspaper').size().reset_index().rename(columns={0:'count'}).head(20))
    print('\nDetail:\n')
    print(BBB.groupby(['newspaper', 'publish_date']).size().rename(columns={0:'count'}).head(20))
    print('')


def save_into_db(database, selected, final_BBB_file):
    is_first = True
    final_BBB = ""
    connection = create_connection(database)
    sql_create_bbb_news_table = """ CREATE TABLE IF NOT EXISTS BBBNews (
                                        id integer PRIMARY KEY,
                                        authors text,
                                        keywords text,
                                        publish_date text,
                                        summary text,
                                        text text,
                                        title text,
                                        top_image text,
                                        url text NOT NULL,
                                        score text,
                                        newspaper text
                                    ); """
    if connection is not None:
        # create projects table
        create_table(connection, sql_create_bbb_news_table)

    else:
        print("Error! cannot create the database connection.")

    with connection:
        df = pd.read_csv(selected)
        for index, row in df.iterrows():
            print(row.url)
            if (check_if_exists(connection, row.url) == 0):
                new = (row.authors, row.keywords, row.publish_date, row.summary, row.text, row.title, row.top_image, row.url, row.score, row.newspaper)
                insert_news(connection, new)
                if is_first:
                    s1 = pd.Series([row.authors, row.keywords, row.publish_date, row.summary, row.text, row.title, row.top_image, row.url, row.score, row.newspaper,])
                    final_BBB = pd.DataFrame([list(s1)], columns=['authors','keywords','publish_date','summary','text','title','top_image','url','score','newspaper'])
                    is_first = False
                else:
                    s1 = pd.Series([row.authors, row.keywords, row.publish_date, row.summary, row.text, row.title, row.top_image, row.url, row.score, row.newspaper])
                    can = pd.DataFrame([list(s1)], columns=['authors','keywords','publish_date','summary','text','title','top_image','url','score','newspaper'])
                    final_BBB = final_BBB.append(can)

    connection.close()

    try:
        final_BBB.to_csv(final_BBB_file, index=False)
        print('\nThese are the news selected:\n')
        print(final_BBB.groupby('newspaper').size().reset_index().rename(columns={0:'count'}))
        print('\nDetail:\n')
        print(final_BBB.groupby(['publish_date', 'newspaper']).size().reset_index().rename(columns={0:'count'}))
        print('')
    except:
        pass


# print('Sending the email to everybody...')
# print('------------------------------------------------------')
# email = EmailSender('amoya@bluecap.com', 'tarra1991')
# email.send_mail(['amoya@bluecap.com'], '%s Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_selection_xgb), 'html')
# email.send_mail(['egilabert@bluecap.com'], '%s XGBOOST Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_selection_xgb), 'html')
