#!/bin/bash

path=$1

if [ -z "${path}" ]; then
	echo "Please specify the path to sync."
	exit 1;
fi

source aws.sh

for f in $path/*/model-gen-latest.h5; do
	d=$(dirname $f)
	./render_video.py --out ${path}/video_0_$(basename $d).json $d
	./render_video.py --allpred --out ${path}/video_1_$(basename $d).json $d
	./render_video.py --allpred --fix-noise --out ${path}/video_2_$(basename $d).json $d
done

./gen_report.py ${path}

aws s3 sync --exclude "*" --include "*.json" --include "*.html" --acl public-read ${path} s3://dance-gan/${path}/

echo "s3 update complete"
