# [BBB Application](https://medium.com/@adriamoyaortiz/classifying-news-articles-to-build-a-customized-newsletter-automatically-ede6ad9ef725)

This Python app runs the following subprocesses:

* Downloading content from articles published in different newspapers during the last 3 days (the urls to each article are previously downloaded by a set of crawlers that run daily on Lambdas). Storing data on S3.

* Text cleaning, tokenization and classification of articles using pre-trained tokenizers and deep neural nets. Storing candidate articles on S3 and RDS.

* Excluding articles that have been included in previous newsletters (RDS).

* Finding text similarities based on tf-idf. Similar articles are clustered together. Article with higher score is parent (and shown first in the newsletter) and similar articles with lower scores are nested below the parent.

* Rendering the newsletter and sending it to subscribers.

This part of the application is triggered once the spiders have crawled all newspapers (running on Lambdas), extracted urls from all articles and stored them into S3. Therefore, it will only work if the right structure is set on AWS. Further, AWS credentials need to be provided to the docker build for testing purposes.

## Directory structure

```shell
# tree . -I __pycache__ --dirsfirst -L 3 > tree.txt
.
├── crawlers/                   # Articles download process
│   ├── article_scraper.py      # Extension of newspaper3k's Article class
│   └── crawlers.py             # Crawlers handler
├── utils/                      # Utils directory
│   ├── logos/
│   ├── allow_stop.py
│   ├── database.py             # Connection with RDS
│   ├── get_date.py
│   ├── mail_body_generator.py  # Newsletter body template
│   ├── select_news.py          # Article classification
│   ├── send_email.py           # Email sender
│   └── similarity.py           # Text similarities
├── Dockerfile
├── README.md
├── download_punkt.py           # Download corpus
├── main.py                     # Main handler
├── requirements.txt            # Requirements
└── run.sh                      # Run script
```

## Testing
<h4>Running dockerized application</h4>

AWS credentials are required.

```shell
# build image
docker build -t ec2_model \
  --build-arg aws_access_key_id=XXX \
  --build-arg aws_secret_access_key=XXX .

# run container
docker run -it --rm --name ec2_model ec2_model
```

In the container run the following command:

```shell
python3 main.py
```

## Manually setting EC2

Deployment package suited for running in EC2 (Ubuntu Server 16.04 LTS (HVM), t2.medium).

<h4>Setting up EC2</h4>

```shell
apt-get -y update
apt-get install -y python3-pip
apt-get install -y python-dev               # ubuntu specific for newspaper3k
apt-get install -y libxml2-dev libxslt-dev  # ubuntu specific for newspaper3k
apt-get install -y curl
pip3 install virtualenv                     # pip install virtualenv
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt            # pip install -r requirements.txt
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3  # ubuntu specific for newspaper3k
```

Donwload nltk corpus

```python
import nltk
nltk.download('punkt')
```

Clone this repository in `/home/ubuntu/`

```shell
git clone https://github.com/adriamoya/bcpnews_ec2.git
```

Set cron job to run the application at reboot event (every time EC2 is run)
```shell
crontab -e  # open crotab editor (file in /tmp/crontab...)
@reboot /home/ubuntu/ec2_model/run.sh  # at reboot execute ~/run.sh
```
