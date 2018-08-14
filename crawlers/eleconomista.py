# -*- coding: utf-8 -*-
import re
import csv
import boto3
import requests
import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from crawlers.article_scraper import ArticleScraper

def parse_eleconomista(crawl_date):
    print("\nInitializing Eleconomista spider ...")
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

    NEWSPAPER = "eleconomista"
    BASE_URL  = "http://www.eleconomista.es/"
    SECTIONS  = ['mercados-cotizaciones', 'economia', 'empresas-finanzas', 'tecnologia']

    try:
        if isinstance(crawl_date, datetime.datetime): # check if argument is datetime.datetime
            start_urls_list = []
            for section in SECTIONS:
                start_urls_list.append( BASE_URL + section + "/")
                print(BASE_URL + section + "/")
    except TypeError:
        print("\nArgument type not valid.")
        pass

    articles_obj = []
    for url in start_urls_list:
        result = requests.get(url)
        c = result.content
        soup = BeautifulSoup(c, "lxml")
        cols = soup.find_all("div", {"class": re.compile("cols")})
        for col in cols:
            titles = col.find_all("h1", {"itemprop": "headline"})
            for title in titles:
                a = title.find_all("a")[0]
                if a:
                    url = a.get("href")
                    if 'http' not in url:
                        url = 'http:' + url
                    new_article = ArticleScraper(url, NEWSPAPER)
                    new_article_obj = new_article.parse_article()
                    if new_article_obj:
                        articles_obj.append(new_article_obj)

    keys = articles_obj[0].keys()
    with open('/tmp/eleconomista_articles.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(articles_obj)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    s3.Object(BUCKET_NAME, 'eleconomista_articles.csv').put(Body=open('/tmp/eleconomista_articles.csv', 'rb'))
