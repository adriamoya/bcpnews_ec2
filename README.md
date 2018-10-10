# Bluecap Banking Breakfast

Deployment package suited for running in EC2 (Ubuntu Server 16.04 LTS (HVM), t2.medium).

```shell
sudo apt-get -y update
sudo apt-get install -y python3-pip
sudo apt-get install python-dev  # ubuntu specific for newspaper3k
sudo apt-get install libxml2-dev libxslt-dev  # ubuntu specific for newspaper3k
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3  # ubuntu specific for newspaper3k
pip install virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Don't forget to run ...

```python
import nltk
nltk.download('punkt')
```

Run on reboot

```shell
crontab -e  # open crotab editor (file in /tmp/crontab...)
@reboot /home/ubuntu/ec2_model/run.sh  # at reboot execute ~/run.sh
```
