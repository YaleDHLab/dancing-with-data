#!/bin/bash

source aws.sh

#./gen_report.py gan_search_mariel1_1102
#aws s3 cp gan_search_mariel1_1102/summary.html s3://dance-gan/gan_search_mariel1_1102/summary.html
#aws s3api put-object-acl --acl public-read --bucket dance-gan --key gan_search_mariel1_1102/summary.html

path=gan_search_mariel1_1102

#aws s3 cp render.html s3://dance-gan/${path}/render.html
#aws s3api put-object-acl --acl public-read --bucket dance-gan --key ${path}/render.html

#aws s3 cp ${path}/test_video.json s3://dance-gan/${path}/test_video.json
#aws s3api put-object-acl --acl public-read --bucket dance-gan --key ${path}/test_video.json

#for f in ${path}/*.json; do
#	aws s3 cp $f s3://dance-gan/${f}
#	aws s3api put-object-acl --acl public-read --bucket dance-gan --key $f
#done

aws s3 sync --exclude "*" --include "*.json" --include "*.html" --acl public-read gan_search_mariel1_1102 s3://dance-gan/gan_search_mariel1_1102/
