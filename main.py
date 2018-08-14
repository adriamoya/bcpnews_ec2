# -*- coding: utf-8 -*-
import os
import time
import datetime
import pandas as pd

from crawlers.crawlers import process_all_newspapers
# from utils.allow_stop import possible_stop
# from utils.get_date import get_date_input
# from utils.mail_body_generator import create_email_body
# from utils.select_news import models_fit_predict, save_into_db, select_news
# from utils.send_email import EmailSender
# from utils.similarity import build_similarities

if __name__ == "__main__":

    # Request input date to crawl. Default is today().
    crawl_date = datetime.datetime.strptime('20180814',"%Y%m%d")  # get_date_input()

    # VARIABLES
    # =========

    # Running date.
    fecha = crawl_date.strftime("%Y%m%d")
    # # Predicitions on articles downloaded for crawl_date`.
    # model_results_lasso = 'data/%s_01_results_lasso.csv' % fecha
    # model_results_xgb = 'data/%s_01_results_xgb.csv' % fecha
    # # Selection of articles (prediction > 50%).
    # news_selection_lasso = 'data/%s_02_selected_bbb_lasso.csv' % fecha
    # news_selection_xgb = 'data/%s_02_selected_bbb_xgb.csv' % fecha
    # # Final selection of articles after checking if they exists in database.
    # final_selection_lasso = 'data/%s_03_final_bbb_lasso.csv' % fecha
    # final_selection_xgb = 'data/%s_03_final_bbb_xgb.csv' % fecha
    # # Database location.
    # database = './BBB_database.db'

    # # Crawling newspapers.
    # print('\nCrawling the news...')
    # print('-'*80)
    # crawl_newspapers(crawl_date)
    # print('Done.')
    #
    # Save all articles into a unique JSON (YYYYMMDD_articles.json`).
    print('\nProcessing all the articles and saving them...')
    print('-'*80)
    process_all_newspapers(fecha)
    print('Done.')

    # # Fitting and predicting. Dumping predictions to `model_results_`.
    # print('\nTraining the models and predicting...')
    # print('-'*80)
    # models_fit_predict(crawl_date)
    # print('Done.')
    #
    # # Selection news with score higher than.
    # print('\nSelecting higher score news...')
    # print('-'*80)
    # select_news(model_results_xgb, news_selection_xgb, 0.5)
    # ## select_news(model_results_lasso, news_selection_lasso, 0.5)

    # # Save selected news to database.
    # print('\nSaving new articles into database...')
    # print('-'*80)
    # ## save_into_db(database, news_selection_lasso, final_selection_lasso)
    # save_into_db(database, news_selection_xgb, final_selection_xgb)

    # # Build similarities from news selection.
    # print('\nSimilarities...')
    # print('-'*80)
    # build_similarities(crawl_date, final_selection_xgb, threshold=0.28, verbose=False)
    #
    # # Read password.
    # PASS = open('pass.txt').read()
    #
    # # Send email.
    # final_articles = './data/%s_04_articles.json' % fecha
    # print('\nSending email...')
    # email = EmailSender('bankingbreakfast@bluecap.com', PASS)
    # email.send_mail(['amoya@bluecap.com'], '%s Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_articles), 'html')
    # # email.send_mail(['bluecapglobal@bluecap.com'], '%s Automatic Bluecap Banking Breakfast' % fecha, create_email_body(final_articles), 'html', _bcc=True)
    # print('\nDone.')
