#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train.csv" \
--dev_file "../DeepCE/data/signature_dev.csv" --test_file "../DeepCE/data/signature_test.csv" \
--dropout 0.1 --batch_size 16 --max_epoch 300 --cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" #  > ../DeepCE/output/cellwise_output.txt
