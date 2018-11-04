#!/bin/bash

EPOCHS=768

output_dir=$1
data_file=${2:-data/npy/mariel-1.npy}

if [ -z "${output_dir}" ]; then
	echo "Please specify output dir."
	exit 1;
fi

log_dir=${output_dir}/logs

echo "Creating log dir ${log_dir}"
mkdir -p $log_dir

npred=(1 2 8)
bsize=(128 64 32)
lr_gen=(1e-3 3e-4 1e-4)
lr_disc=(1e-4 3e-5 1e-5 3e-6 1e-6)
rigidity=(1e-4 1e-6)

for bs in "${bsize[@]}"; do
for lrd in "${lr_disc[@]}"; do
for lrg in "${lr_gen[@]}"; do
for rig in "${rigidity[@]}"; do
for np in "${npred[@]}"; do
	if [ $(echo "$lrg<$lrd" | tr e E | bc) -eq 1 ]; then
		continue;
	fi
	echo "wrap_grace.sh --rotate --lr-gen $lrg --lr-disc $lrd --rigidity $rig --batch-size $bs --n-pred $np --epochs $EPOCHS --data $data_file $output_dir"
	sbatch -p gpu --gres gpu:1 -c 2 -o "${log_dir}/slurm-%j.out" \
		wrap_grace.sh --do-plots --rotate --lr-gen $lrg --lr-disc $lrd --rigidity $rig --batch-size $bs --n-pred $np --epochs $EPOCHS --data $data_file $output_dir
done
done
done
done
done
