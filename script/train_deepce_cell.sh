#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train_cell_2.csv" \
--dev_file "../DeepCE/data/signature_dev_cell_2.csv" --test_file "../DeepCE/data/signature_test_cell_2.csv" \
--dropout 0.3 --batch_size 16 --max_epoch 200 --unfreeze_steps 0,0,0,0 \
--cell_ge_file "../DeepCE/data/gene_expression_combat_norm_978" # > ../DeepCE/output/cellwise_output_ran5.txt
