#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python ../MultiDCP/ehill_multidcp_pretrain.py \
--drug_file "../MultiDCP/data/all_drugs_l1000.csv" \
--gene_file "../MultiDCP/data/gene_vector.csv"  --hill_train_file "../MultiDCP/data/ehill_data/high_confident_data_train.csv" \
--hill_dev_file "../MultiDCP/data/ehill_data/high_confident_data_dev.csv" \
--hill_test_file "../MultiDCP/data/ehill_data/high_confident_data_test.csv" \
--train_file "../MultiDCP/data/pert_transcriptom/signature_train_cell_2.csv" \
--dev_file "../MultiDCP/data/pert_transcriptom/signature_dev_cell_2.csv" \
--test_file "../MultiDCP/data/pert_transcriptom/signature_test_cell_2.csv" \
--dropout 0.1 --batch_size 64 --max_epoch 100 \
--all_cells "../MultiDCP/data/ehill_data/pretrain_cell_list_ehill.p" \
--cell_ge_file "../MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv" \
--linear_encoder_flag # > ../MultiDCP/output/cellwise_output_ran5.txt
