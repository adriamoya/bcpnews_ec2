# -*- coding: utf-8 -*-
import json
import boto3
import datetime
from newspaper import Article

class ArticleScraper(Article):

    """ For a given article url, it downloads and parses some specific data and writes a JSON in the output_file """

    def __init__(self, url, newspaper):
        """ Initialize ArticleScraper """
        self.article_obj = {}
        self.article_obj["url"] = url
        self.article_obj["newspaper"] = newspaper

        if self.article_obj:
            # initiate article
            self.article = Article(url, language="es")
            # parse article
            # self.parse_article()

    def parse_article(self):
        """ Download, Parse and NLP a given article """
        # try:
        # download source code
        self.article.download()

        # parse code
        self.article.parse()

        # populate article obj with parsed data
        try:
            self.article_obj["title"] = self.article.title
            # self.article_obj["title"] = self.article.title.encode("utf-8").strip()
        except:
            self.article_obj["title"] = ""

        try:
            self.article_obj["publish_date"] = self.article.publish_date
            # self.article_obj["publish_date"] = self.article.publish_date.encode("utf-8").strip()
        except:
            self.article_obj["publish_date"] = ""

        try:
            self.article_obj["text"] = self.article.text
            # self.article_obj["text"] = self.article.text.encode("utf-8").strip()
        except:
            self.article_obj["text"] = ""

        try:
            self.article_obj["top_image"] = self.article.top_image
        except:
            self.article_obj["top_image"] = ""

        self.article.nlp()

        try:
            self.article_obj["summary"] = self.article.summary
        except:
            self.article_obj["summary"] = ""

        try:
            self.article_obj["keywords"] = self.article.keywords
        except:
            self.article_obj["keywords"] = []

        print(self.article_obj)
        return self.article_obj

        # except:
        #     pass
