#!/usr/bin/env bash



python /raid/home/yoyowu/MultiDCP/MultiDCP/multidcp_ae.py --device 0 \
--exp 'for_binary_0913' \
--drug_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/all_drugs_l1000.csv" \
--gene_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/gene_vector.csv"  --train_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/pert_transcriptom/signature_train_cell_1.csv" \
--dev_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/pert_transcriptom/signature_dev_cell_1.csv" --test_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/pert_transcriptom/signature_test_cell_1.csv" \
--dropout 0.3 --batch_size 64 --max_epoch 500 \
--ae_input_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4_train.csv" \
--ae_label_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4_train.csv" \
--predicted_result_for_testset "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/teacher_student/teach_stu_perturbedGX.csv" \
--hidden_repr_result_for_testset "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/teacher_student/teach_stu_perturbedGX_hidden.csv" \
--cell_ge_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/adjusted_ccle_tcga_ad_tpm_log2.csv" \
--all_cells "/raid/home/yoyowu/MultiDCP/MultiDCP_data//data/ccle_tcga_ad_cells.p" \
> MultiDCP/Sep_outlogs/0913_for_binary.log 2>&1 &
echo "done"
# > ../MultiDCP/output/cellwise_output_ran5.txt
