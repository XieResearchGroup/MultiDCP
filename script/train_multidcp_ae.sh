#!/usr/bin/env bash
exp=0926_binary_split1_down
python /raid/home/yoyowu/MultiDCP/MultiDCP/multidcp_ae.py --device 5 \
--exp ${exp} \
--model_save_path "/raid/home/yoyowu/MultiDCP/saved_models/${exp}.pt" \
--drug_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/all_drugs_l1000.csv" \
--gene_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/gene_vector.csv"  --train_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_train_1.csv" \
--dev_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_dev_1.csv"  --test_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_test_1.csv"  \
--dropout 0.3 --batch_size 64 --max_epoch 500 \
--ae_input_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4" \
--ae_label_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4" \
--predicted_result_for_testset "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/teacher_student/teach_stu_perturbedGX.csv" \
--hidden_repr_result_for_testset "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/teacher_student/teach_stu_perturbedGX_hidden.csv" \
--cell_ge_file "/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/adjusted_ccle_tcga_ad_tpm_log2.csv" \
--all_cells "/raid/home/yoyowu/MultiDCP/MultiDCP_data//data/ccle_tcga_ad_cells.p" \
> MultiDCP/Sep_outlogs/${exp}.log 2>&1 &
echo "done"
# > ../MultiDCP/output/cellwise_output_ran5.txt
