# -*- coding: utf-8 -*-
import os
import json
import time
import datetime
import pandas as pd

from crawlers.crawlers import process_all_newspapers
from utils.select_news import models_fit_predict, select_news, save_into_db
from utils.mail_body_generator import create_email_body
from utils.send_email import EmailSender
from utils.similarity import build_similarities

if __name__ == "__main__":

    # Request input date to crawl. Default is today().
    crawl_date = datetime.datetime.now()
    fecha = crawl_date.strftime("%Y%m%d")

    # Save all articles into a unique JSON (YYYYMMDD_articles.json`).
    print('\nProcessing all the articles and saving them...')
    print('-'*80)
    df = process_all_newspapers(fecha)
    print('Done.')

    # Fitting and predicting. Dumping predictions to `model_results_`.
    print('\nTraining the models and predicting...')
    print('-'*80)
    df = models_fit_predict(fecha, df)
    print('Done.')

    # Selection news with score higher than.
    print('\nSelecting higher score news...')
    print('-'*80)
    df = select_news(df, fecha, threshold=0.5)
    print('Done.')

    # Save selected news to database.
    print('\nSaving new articles into database...')
    print('-'*80)
    with open('rds_params.json') as f:
        database = json.load(f)
    df = save_into_db(df, fecha, database)

    # Build similarities from news selection.
    print('\nSimilarities...')
    print('-'*80)
    final_articles = build_similarities(df, fecha, threshold=0.30, verbose=False)

    # Read password.
    PASS = open('pass.txt').read()

    # Send email.
    # final_articles = './data/%s_04_articles.json' % fecha
    print('\nSending email...')
    print('-'*80)
    email = EmailSender('bankingbreakfast@bluecap.com', PASS)
    email.send_mail(['amoya@bluecap.com', 'cmoyag4@gmail.com'], '%s Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_articles), 'html')
    # email.send_mail(['bluecapglobal@bluecap.com'], '%s Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_articles), 'html', _bcc=True)
    print('\nDone.')
