import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
from datetime import datetime
import torch
from torch import save
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/models')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils')
import multidcp
import datareader
import metric
import wandb
import pdb
import pickle
from scheduler_lr import step_lr
from loss_utils import apply_NodeHomophily
from tqdm import tqdm
from multidcp_ae_utils import *
import warnings
warnings.filterwarnings("ignore")

USE_WANDB = False
if USE_WANDB:
    wandb.init(project="MultiDCP_AE_loss")
else:
    os.environ["WANDB_MODE"] = "dryrun"

# check cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

if __name__ == '__main__':
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='MultiDCP AE')
    parser.add_argument('--drug_file')
    parser.add_argument('--gene_file')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--test_file')
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--ae_input_file')
    parser.add_argument('--ae_label_file')
    parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file')
    parser.add_argument('--max_epoch', type = int)
    parser.add_argument('--predicted_result_for_testset', help = "the file directory to save the predicted test dataframe")
    parser.add_argument('--hidden_repr_result_for_testset', help = "the file directory to save the test data hidden representation dataframe")
    parser.add_argument('--all_cells')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--linear_encoder_flag', dest = 'linear_encoder_flag', action='store_true', default=False,
                        help = 'whether the cell embedding layer only have linear layers')

    args = parser.parse_args()

    all_cells = list(pickle.load(open(args.all_cells, 'rb')))
    DATA_FILTER = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422','BRD-U01690642','BRD-U08759356','BRD-U25771771', 'BRD-U33728988', 'BRD-U37049823',
                'BRD-U44618005', 'BRD-U44700465','BRD-U51951544', 'BRD-U66370498','BRD-U68942961', 'BRD-U73238814',
                'BRD-U82589721','BRD-U86922168','BRD-U97083655'],
            "pert_type": ["trt_cp"],
            "cell_id": all_cells,# ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC'],
            "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}


    ae_data = datareader.AEDataLoader(device, args)
    data = datareader.PerturbedDataLoader(DATA_FILTER, device, args)
    ae_data.setup()
    data.setup()
    print('#Train: %d' % len(data.train_data))
    print('#Dev: %d' % len(data.dev_data))
    print('#Test: %d' % len(data.test_data))
    print('#Train AE: %d' % len(ae_data.train_data))
    print('#Dev AE: %d' % len(ae_data.dev_data))
    print('#Test AE: %d' % len(ae_data.test_data))

    # parameters initialization
    model_param_registry = initialize_model_registry()
    model_param_registry.update({'num_gene': np.shape(data.gene)[0],
                                'pert_idose_input_dim': len(DATA_FILTER['pert_idose']),
                                'dropout': args.dropout, 
                                'linear_encoder_flag': args.linear_encoder_flag})

    # model creation
    print('--------------with linear encoder: {0!r}--------------'.format(args.linear_encoder_flag))
    model = multidcp.MultiDCP_AE(device=device, model_param_registry=model_param_registry)
    model.init_weights(pretrained = False)
    model.to(device)
    model = model.double() 

    if USE_WANDB:
        wandb.watch(model, log="all")

    # training
    metrics_summary = defaultdict(
        pearson_list_ae_dev = [],
        pearson_list_ae_test = [],
        pearson_list_perturbed_dev = [],
        pearson_list_perturbed_test = [],
        spearman_list_ae_dev = [],
        spearman_list_ae_test = [],
        spearman_list_perturbed_dev = [],
        spearman_list_perturbed_test = [],
        rmse_list_ae_dev = [],
        rmse_list_ae_test = [],
        rmse_list_perturbed_dev = [],
        rmse_list_perturbed_test = [],
        precisionk_list_ae_dev = [],
        precisionk_list_ae_test = [],
        precisionk_list_perturbed_dev = [],
        precisionk_list_perturbed_test = [],
    )

    model.load_state_dict(torch.load('best_multidcp_ae_model.pt', map_location = device))
    # pdb.set_trace()
    data_save = True

    epoch_loss = 0
    lb_np_ls = []
    predict_np_ls = []
    hidden_np_ls = []
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(tqdm(data.test_dataloader())):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            predict, cells_hidden_repr = model(input_cell_gex=cell_feature, input_drug = drug, 
                                            input_gene = data.gene, mask = mask,
                                            input_pert_idose = pert_idose, job_id = 'perturbed')
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np_ls.append(lb.cpu().numpy()) 
            predict_np_ls.append(predict.cpu().numpy()) 
            hidden_np_ls.append(cells_hidden_repr.cpu().numpy()) 

        lb_np = np.concatenate(lb_np_ls, axis = 0)
        predict_np = np.concatenate(predict_np_ls, axis = 0)
        hidden_np = np.concatenate(hidden_np_ls, axis = 0)
        if data_save:
            genes_cols = sorted_test_input.columns[5:]
            assert sorted_test_input.shape[0] == predict_np.shape[0]
            predict_df = pd.DataFrame(predict_np, index = sorted_test_input.index, columns = genes_cols)
            # hidden_df = pd.DataFrame(hidden_np, index = sorted_test_input.index, columns = [x for x in range(50)])
            ground_truth_df = pd.DataFrame(lb_np, index = sorted_test_input.index, columns = genes_cols)
            result_df  = pd.concat([sorted_test_input.iloc[:, :5], predict_df], axis = 1)
            ground_truth_df = pd.concat([sorted_test_input.iloc[:,:5], ground_truth_df], axis = 1)
            # hidden_df = pd.concat([sorted_test_input.iloc[:,:5], hidden_df], axis = 1) 
                    
            print("=====================================write out data=====================================")
            result_df.loc[[x for x in range(len(result_df))],:].to_csv(predicted_result_for_testset, index = False)
            # hidden_df.loc[[x for x in range(len(hidden_df))],:].to_csv(hidden_repr_result_for_testset, index = False)
            # ground_truth_df.loc[[x for x in range(len(result_df))],:].to_csv('../MultiDCP/data/side_effect/test_for_same.csv', index = False)

        test_epoch_end(epoch_loss = epoch_loss, lb_np = lb_np, 
                        predict_np = predict_np, steps_per_epoch = i+1, 
                        epoch = epoch, metrics_summary = metrics_summary,
                        job = 'perturbed', USE_WANDB = USE_WANDB)

end_time = datetime.now()
print(end_time - start_time)