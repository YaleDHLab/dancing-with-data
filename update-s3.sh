#!/bin/bash

source aws.sh

./gen_report.py gan_search_mariel1_1102
aws s3 cp gan_search_mariel1_1102/summary.html s3://dance-gan/gan_search_mariel1_1102/summary.html
aws s3api put-object-acl --acl public-read --bucket dance-gan --key gan_search_mariel1_1102/summary.html

aws s3 cp render.html s3://dance-gan/gan_search_mariel1_1102/render.html
aws s3api put-object-acl --acl public-read --bucket dance-gan --key gan_search_mariel1_1102/render.html

#aws s3 sync --exclude "*" --include "*.html" --include "*.png" --acl public-read gan_search_mariel1_1102 s3://dance-gan/gan_search_mariel1_1102/
#aws s3 sync --exclude "*" --include "*.html" --acl public-read gan_search_mariel1_1102 s3://dance-gan/gan_search_mariel1_1102/
