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
from multidcp_ae_utils import *

USE_wandb = True
if USE_wandb:
    wandb.init(project="MultiDCP_AE_loss")
else:
    os.environ["WANDB_MODE"] = "dryrun"

# check cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

def model_training(args, model, data, ae_data, metrics_summary):

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_dev_pearson = float("-inf")

    for epoch in range(max_epoch):
        
        print("Iteration %d:" % (epoch))
        print_lr(optimizer)
        model.train()
        data_save = False
        epoch_loss = 0

        for i, (feature, label, _) in enumerate(ae_data.train_dataloader()):

            optimizer.zero_grad()
            #### the auto encoder step doesn't need other input rather than feature
            predict, cell_hidden_ = model(input_cell_gex=feature, job_id = 'ae', epoch = epoch)
            loss_t = model.loss(label, predict)
            loss_t.backward()
            optimizer.step()
            epoch_loss += loss_t.item()

        print('AE Train loss:')
        print(epoch_loss/(i+1))
        if USE_WANDB:
            wandb.log({'AE Train loss': epoch_loss/(i+1)}, step = epoch)

        model.eval()
        epoch_loss = 0
        lb_np = np.empty([0, 978])
        predict_np = np.empty([0, 978])
        with torch.no_grad():
            for i, (feature, label, _) in enumerate(ae_data.val_dataloader()):
                predict, _ = model(input_cell_gex=feature, job_id = 'ae', epoch = epoch)
                loss = model.loss(label, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            validation_epoch_end(epoch_loss = epoch_loss, lb_np = lb_np, 
                                predict_np = predict_np, steps_per_epoch = i+1, 
                                epoch = epoch, metrics_summary = metrics_summary,
                                job = 'ae', USE_WANDB = USE_WANDB)

            if best_dev_pearson < metrics_summary['pearson_list_ae_dev'][-1]:
                best_dev_pearson = metrics_summary['pearson_list_ae_dev'][-1]
                save(model.sub_multidcp.state_dict(), 'best_sub_multidcp_storage_split3')

        epoch_loss = 0
        lb_np = np.empty([0, 978])
        predict_np = np.empty([0, 978])
        hidden_np = np.empty([0, 50])
        with torch.no_grad():
            for i, (feature, label, _) in enumerate(ae_data.test_dataloader()):
                predict, hidden = model(input_cell_gex=feature, job_id = 'ae')
                loss = model.loss(label, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
                hidden_np = np.concatenate((hidden_np, hidden.cpu().numpy()), axis=0)

            if data_save:
                test_ae_label_file = pd.read_csv(args.ae_label_file + '_test.csv', index_col=0)
                hidden_df = pd.DataFrame(hidden_np, index = list(test_ae_label_file.index), columns = [x for x in range(50)])
                print('++++++++++++++++++++++++++++Write hidden state out++++++++++++++++++++++++++++++++')
                hidden_df.to_csv(args.hidden_repr_result_for_testset)

            test_epoch_end(epoch_loss = epoch_loss, lb_np = lb_np, 
                                predict_np = predict_np, steps_per_epoch = i+1, 
                                epoch = epoch, metrics_summary = metrics_summary,
                                job = 'ae', USE_WANDB = USE_WANDB)

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

    model_training(args, model, data, ae_data, metrics_summary)
    report_final_results(metrics_summary, ae = True)
    end_time = datetime.now()
    print(end_time - start_time)
