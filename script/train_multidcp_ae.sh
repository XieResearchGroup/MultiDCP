#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python ../MultiDCP/multidcp_ae.py --drug_file "../MultiDCP/data/all_drugs_l1000.csv" \
--gene_file "../MultiDCP/data/gene_vector.csv"  --train_file "../MultiDCP/data/pert_transcriptom/signature_train_cell_1.csv" \
--dev_file "../MultiDCP/data/pert_transcriptom/signature_dev_cell_1.csv" --test_file "../MultiDCP/data/pert_transcriptom/signature_test_cell_1.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 500 \
--ae_input_file "../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4" \
--ae_label_file "../MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4" \
--cell_ge_file "../MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv" \
--all_cells "../MultiDCP/data/ccle_tcga_ad_cells.p" \
 > /raid/home/yoyowu/MultiDCP/MultiDCP/outlogs/0423issue2.log 2>&1 & 
