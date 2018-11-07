#!/bin/bash

path=$1

if [ -z "${path}" ]; then
	echo "Please specify path to report."
	exit
fi

source aws.sh

./gen_report.py ${path}
aws s3 cp ${path}/summary.html s3://dance-gan/${path}/summary.html
aws s3api put-object-acl --acl public-read --bucket dance-gan --key ${path}/summary.html

aws s3 cp render.html s3://dance-gan/${path}/render.html
aws s3api put-object-acl --acl public-read --bucket dance-gan --key ${path}/render.html

#aws s3 sync --exclude "*" --include "*.html" --include "*.png" --acl public-read gan_search_mariel1_1102 s3://dance-gan/gan_search_mariel1_1102/
#aws s3 sync --exclude "*" --include "*.html" --acl public-read gan_search_mariel1_1102 s3://dance-gan/gan_search_mariel1_1102/
