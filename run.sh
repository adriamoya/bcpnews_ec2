#!/bin/bash

# Printing date
now="$(date)"
echo "--> Starting $now"

# Running application
cd /home/ubuntu/ec2_model
stdbuf -oL /home/ubuntu/ec2_model/env/bin/python /home/ubuntu/ec2_model/main.py > log
