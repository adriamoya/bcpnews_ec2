# -*- coding: utf-8 -*-
import csv
import boto3
import requests
import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from crawlers.article_scraper import ArticleScraper

def parse_expansion(crawl_date):
    print("\nInitializing Expansion spider ...")
    print("-"*80)
    # connection to s3 bucket
    BUCKET_NAME = "bluecaparticles"
    s3 = boto3.client("s3")
    try:
        bucket = s3.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": "eu-central-1"}
            )
        print("Bucket created")
    except Exception as e:
        print("Bucket already exists")

    NEWSPAPER = 'expansion'
    BASE_URL  = 'http://www.expansion.com/hemeroteca/'

    try:
        if isinstance(crawl_date, datetime.datetime): # check if argument is datetime.datetime
            dates = [crawl_date - datetime.timedelta(days=3), crawl_date - datetime.timedelta(days=2), crawl_date - datetime.timedelta(days=1), crawl_date]
            sections = ['apertura', 'media', 'cierre', 'noche']
            start_urls_list = []
            for date in dates:
                for section in sections:
                    start_urls_list.append(BASE_URL + date.strftime("%Y/%m/%d") + "/" + section + ".html")
    except TypeError:
        print("\nArgument type not valid.")
        pass

    articles_obj = []
    for url in start_urls_list:
        result = requests.get(url)
        c = result.content
        soup = BeautifulSoup(c, "lxml")
        articles = soup.find_all("article")
        for article in articles:
            titles = article.find_all('h2')
            for title in titles:
                a = title.find_all('a')[0]
                url = a.get('href')
                if url:
                    new_article = ArticleScraper(url, NEWSPAPER)
                    new_article_obj = new_article.parse_article()
                    if new_article_obj:
                        articles_obj.append(new_article_obj)

    keys = articles_obj[0].keys()
    with open('/tmp/expansion_articles.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(articles_obj)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    s3.Object(BUCKET_NAME, 'expansion_articles.csv').put(Body=open('/tmp/expansion_articles.csv', 'rb'))
