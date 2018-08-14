# -*- coding: utf-8 -*-
import os
import csv
import boto3
import pandas as pd
from crawlers.article_scraper import ArticleScraper

def process_all_newspapers(crawl_date):

    S3_BUCKET = "bluecaparticles"

    # Connection to s3 bucket
    BUCKET_NAME = "bluecaparticles"
    client = boto3.client("s3")

    try:
        bucket = client.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": "eu-central-1"}
            )
        print("Bucket created")
    except Exception as e:
        print("Bucket already exists")

    # Create output directory if not exists.
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read downloaded urls
    urls_downloaded = []
    path_articles = [x['Key'] for x in client.list_objects(Bucket=S3_BUCKET)['Contents'] if 'articles' in x['Key']]
    for idx, path_newspaper_articles in enumerate(path_articles):
        if idx == 0:
            newspaper_articles = client.get_object(Bucket=S3_BUCKET, Key=path_newspaper_articles)
            print(idx, path_newspaper_articles)
            df_temp = pd.read_csv(newspaper_articles['Body'], encoding='utf8')
            urls_downloaded.append(df_temp)
    del df_temp
    df_subm = pd.concat(urls_downloaded)
    print(df_subm.shape)

    # Download articles
    articles_obj = []
    for idx, row in df_subm.iterrows():
        url = row['url']
        print(url)
        newspaper = row['newspaper']
        new_article = ArticleScraper(url, newspaper)
        new_article_obj = new_article.parse_article()
        if new_article_obj:
            articles_obj.append(new_article_obj)

    # Write articles to tmp and push to s3
    keys = articles_obj[0].keys()
    with open('output/%s_articles.csv' % crawl_date, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(articles_obj)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    s3.Object(BUCKET_NAME, 'output/%s_articles.csv' % crawl_date).put(Body=open('/output/%s_articles.csv' % crawl_date, 'rb'))
