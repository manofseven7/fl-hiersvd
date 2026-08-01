#!/usr/bin/env bash
set -e
SEEDS="42 43 44 45 46"
STD="fedavg fedprox scaffold fednova fedkd qsgd fedpaq fedcom topk adaptive_topk sign powersgd"

for ds in fashion har; do
  for m in $STD; do
    for s in $SEEDS; do python run.py --method $m --dataset $ds --seed $s; done
  done
  # Proposed WITHOUT feature selection (unmasked image path on Fashion; ablation baseline on HAR)
  for s in $SEEDS; do python run.py --method proposed --dataset $ds --seed $s --tag nofs --fs off; done
done

# Proposed WITH feature selection (HAR only)
for s in $SEEDS; do python run.py --method proposed --dataset har --seed $s --tag fs --fs on; done

# CIFAR-10 matched protocol (100 clients / 400 rounds)
for m in fedavg powersgd qsgd proposed; do
  for s in $SEEDS; do python run.py --method $m --dataset cifar --seed $s; done
done

echo "Runs done -> build tables with scripts/make_tables.py"