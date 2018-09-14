# -*- coding: utf-8 -*-
import os
import csv
import boto3
import pandas as pd
from io import StringIO
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
    print('\n--> Reading downloaded urls...')
    urls_downloaded = []
    path_articles = [x['Key'] for x in client.list_objects(Bucket=S3_BUCKET)['Contents'] if 'urls' in x['Key']]
    for idx, path_newspaper_articles in enumerate(path_articles):
        newspaper_articles = client.get_object(Bucket=S3_BUCKET, Key=path_newspaper_articles)
        print(idx, path_newspaper_articles)
        df_temp = pd.read_csv(newspaper_articles['Body'], encoding='utf8')
        print(df_temp.shape)
        urls_downloaded.append(df_temp)
    del df_temp
    df_subm = pd.concat(urls_downloaded)
    # df_subm = pd.concat(urls_downloaded, sort=True)
    print(df_subm.shape)

    # Download articles
    print('\n--> Downloading articles...')
    articles_obj = []
    for idx, row in df_subm.iterrows():
        url = row['url']
        newspaper = row['newspaper']
        timestamp = row['timestamp']
        print(url)
        new_article = ArticleScraper(url, timestamp, newspaper)
        new_article_obj = new_article.parse_article()
        if new_article_obj:
            articles_obj.append(new_article_obj)

    # Saving to s3
    print('\n--> Saving to s3...')
    df = pd.DataFrame(articles_obj)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET_NAME, 'output/%s_articles.csv' % crawl_date).put(Body=csv_buffer.getvalue())

    return df
