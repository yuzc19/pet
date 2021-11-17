for seed in 13 21 42 87 100
do
sbatch -p rtx2080 --job-name exp-pet-SST-2  --output=slurm/%j-$seed.out -N 1 -n 1 --cpus-per-task=8 --mem=15G --gres=gpu:1  <<EOF
#!/bin/sh
TASK=SST-2 \
SEED=$seed \
PATTERN_IDS=0 \
bash run_experiment.sh
EOF
done