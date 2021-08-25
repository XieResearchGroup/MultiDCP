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
from collections import defaultdict

USE_WANDB = True
PRECISION_DEGREE = [10, 20, 50, 100]
if USE_WANDB:
    wandb.init(project="MultiDCP_AE_ehill")
else:
    os.environ["WANDB_MODE"] = "dryrun"

# check cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

def initialize_model_registry():

    model_param_registry = defaultdict(
        drug_input_dim = {'atom': 62, 'bond': 6},
        drug_emb_dim = 128,
        conv_size = [16, 16],
        degree = [0, 1, 2, 3, 4, 5],
        gene_emb_dim = 128,
        gene_input_dim = 128,
        cell_id_input_dim = 978,
        cell_feature_emb_dim = 32,
        pert_idose_emb_dim = 4,
        hid_dim = 128,
        num_gene = 978,
        loss_type = 'point_wise_mse', #'point_wise_mse' # 'list_wise_ndcg'
        initializer = torch.nn.init.kaiming_uniform_
    )
    return model_param_registry

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("============current learning rate is {0!r}".format(param_group['lr']))

def validation_epoch_end(epoch_loss, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary):

    print('Perturbed gene expression profile Dev loss:')
    print(epoch_loss / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Dev loss': epoch_loss/steps_per_epoch}, step=epoch)
    
    rmse = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_perturbed_dev'].append(rmse)
    print('Perturbed gene expression profile RMSE: %.4f' % rmse)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Dev RMSE': rmse}, step=epoch)
    pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_list_perturbed_dev'].append(pearson)
    print('Perturbed gene expression profile Pearson\'s correlation: %.4f' % pearson)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Dev Pearson': pearson}, step = epoch)
    spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_list_perturbed_dev'].append(spearman)
    print('Perturbed gene expression profile Spearman\'s correlation: %.4f' % spearman)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Dev Spearman': spearman}, step = epoch)
    perturbed_precision = []
    for k in PRECISION_DEGREE:
        precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
        print("Perturbed gene expression profile Precision@%d Positive: %.4f" % (k, precision_pos))
        print("Perturbed gene expression profile Precision@%d Negative: %.4f" % (k, precision_neg))
        perturbed_precision.append([precision_pos, precision_neg])
    metrics_summary['precisionk_list_perturbed_dev'].append(perturbed_precision)

def test_epoch_end(epoch_loss, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary):
    print('Perturbed gene expression profile Test loss:')
    print(epoch_loss / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Test Loss': epoch_loss / steps_per_epoch}, step=epoch)
    rmse = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_perturbed_test'].append(rmse)
    print('Perturbed gene expression profile RMSE: %.4f' % rmse)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Test RMSE': rmse}, step=epoch)
    pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_list_perturbed_test'].append(pearson)
    print('Perturbed gene expression profile Pearson\'s correlation: %.4f' % pearson)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Test Pearson': pearson}, step=epoch)
    spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_list_perturbed_test'].append(spearman)
    print('Perturbed gene expression profile Spearman\'s correlation: %.4f' % spearman)
    if USE_WANDB:
        wandb.log({'Perturbed gene expression profile Test Spearman': spearman}, step=epoch)
    perturbed_precision_test = []
    for k in PRECISION_DEGREE:
        precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
        print("Perturbed gene expression profile Precision@%d Positive: %.4f" % (k, precision_pos))
        print("Perturbed gene expression profile Precision@%d Negative: %.4f" % (k, precision_neg))
        perturbed_precision_test.append([precision_pos, precision_neg])
    metrics_summary['precisionk_list_perturbed_test'].append(perturbed_precision_test)

def report_final_results(metrics_summary):

    best_dev_epoch = np.argmax(metrics_summary['pearson_list_perturbed_dev'])
    print("Epoch %d got best Perturbed Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['pearson_list_perturbed_dev'][best_dev_epoch]))
    print("Epoch %d got Perturbed Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['spearman_list_perturbed_dev'][best_dev_epoch]))
    print("Epoch %d got Perturbed RMSE on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['rmse_list_perturbed_dev'][best_dev_epoch]))
    print("Epoch %d got Perturbed P@100 POS and NEG on dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                    metrics_summary['precisionk_list_perturbed_dev'][best_dev_epoch][-1][0],
                                                                    metrics_summary['precisionk_list_perturbed_dev'][best_dev_epoch][-1][1]))

    print("Epoch %d got Perturbed Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['pearson_list_perturbed_test'][best_dev_epoch]))
    print("Epoch %d got Perturbed Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['spearman_list_perturbed_test'][best_dev_epoch]))
    print("Epoch %d got Perturbed RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['rmse_list_perturbed_test'][best_dev_epoch]))
    print("Epoch %d got Perturbed P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                    metrics_summary['precisionk_list_perturbed_test'][best_dev_epoch][-1][0],
                                                                    metrics_summary['precisionk_list_perturbed_test'][best_dev_epoch][-1][1]))

    best_test_epoch = np.argmax(metrics_summary['pearson_list_perturbed_test'])
    print("Epoch %d got Perturbed best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, metrics_summary['pearson_list_perturbed_test'][best_test_epoch]))
    print("Epoch %d got Perturbed Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, metrics_summary['spearman_list_perturbed_test'][best_test_epoch]))
    print("Epoch %d got Perturbed RMSE on test set: %.4f" % (best_test_epoch + 1, metrics_summary['rmse_list_perturbed_test'][best_test_epoch]))
    print("Epoch %d got Perturbed P@100 POS and NEG on test set: %.4f, %.4f" % (best_test_epoch + 1,
                                                                    metrics_summary['precisionk_list_perturbed_test'][best_test_epoch][-1][0],
                                                                    metrics_summary['precisionk_list_perturbed_test'][best_test_epoch][-1][1]))

def model_training(args, model, data, hill_data, metrics_summary):

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_dev_pearson = float("-inf")

    for epoch in range(args.max_epoch):

        print("Iteration %d:" % (epoch+1))
        print_lr(optimizer)
        model.train()
        epoch_loss = 0

        for i, (ft, lb, cell_type) in enumerate(data.train_dataloader()):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            optimizer.zero_grad()
            predict, cell_hidden_ = model(drug, data.gene.to(device), mask, cell_feature, pert_idose,
                                job_id='perturbed', epoch=epoch)
            loss_t = model.loss(lb, predict)
            loss_t.backward()
            optimizer.step()
            if i == 1:
                print('__________________________input__________________________')
                print(cell_feature)
                print('__________________________hidden__________________________')
                print(cell_hidden_)
            epoch_loss += loss.item()
        print('Perturbed gene expression profile Train loss:')
        print(epoch_loss/(i+1))
        if USE_WANDB:
            wandb.log({'Perturbed gene expression profile Train loss': epoch_loss/(i+1)}, step = epoch)

        model.eval()

        epoch_loss = 0
        lb_np = np.empty([0, 978])
        predict_np = np.empty([0, 978])
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(data.val_dataloader()):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(drug, data.gene, mask, cell_feature, pert_idose,
                                job_id='perturbed', epoch = epoch)
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            
            validation_epoch_end(epoch_loss, lb_np, predict_np, i+1, epoch, metrics_summary)

            if best_dev_pearson < metrics_summary['pearson_list_perturbed_dev'][-1]:
                best_dev_pearson = metrics_summary['pearson_list_perturbed_dev'][-1]

        epoch_loss = 0
        lb_np = np.empty([0, 978])
        predict_np = np.empty([0, 978])
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(data.test_dataloader()):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(drug, data.gene, mask, cell_feature, pert_idose,
                                job_id='perturbed', epoch = 0)
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

            test_epoch_end(epoch_loss, lb_np, predict_np, i+1, epoch, metrics_summary)


if __name__ == '__main__':

    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='MultiDCP Ehill finetuning')
    parser.add_argument('--drug_file')
    parser.add_argument('--gene_file')
    parser.add_argument('--dropout', type = float)
    parser.add_argument('--hill_train_file')
    parser.add_argument('--hill_dev_file')
    parser.add_argument('--hill_test_file')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--test_file')
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--max_epoch', type = int)
    parser.add_argument('--all_cells')
    parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file')
    parser.add_argument('--linear_encoder_flag', dest = 'linear_encoder_flag', action='store_true', default=False,
                        help = 'whether the cell embedding layer only have linear layers')

    args = parser.parse_args()

    all_cells = list(pickle.load(open(args.all_cells, 'rb')))
    DATA_FILTER = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
            # "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
            "cell_id": all_cells,
            "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

    hill_data = datareader.EhillDataLoader(DATA_FILTER, device, args)
    data = datareader.PerturbedDataLoader(DATA_FILTER, device, args)
    hill_data.setup()
    data.setup()                             
    print('#Train hill data: %d' % len(hill_data.train_data))
    print('#Dev hill data: %d' % len(hill_data.dev_data))
    print('#Test hill data: %d' % len(hill_data.test_data))
    print('#Train perturbed data: %d' % len(data.train_data))
    print('#Dev perturbed data: %d' % len(data.dev_data))
    print('#Test perturbed data: %d' % len(data.test_data))

    # parameters initialization
    model_param_registry = initialize_model_registry()
    model_param_registry.update({'num_gene': np.shape(data.gene)[0],
                                'pert_idose_input_dim': len(DATA_FILTER['pert_idose']),
                                'dropout': args.dropout, 
                                'linear_encoder_flag': args.linear_encoder_flag})

    # model creation
    print('--------------with linear encoder: {0!r}--------------'.format(args.linear_encoder_flag))
    model = multidcp.MultiDCPEhillPretraining(device=device, model_param_registry=model_param_registry)
    model.init_weights(pretrained = True)
    model.to(device)
    model = model.double()

    if USE_WANDB:
        wandb.watch(model, log="all")

    metrics_summary = defaultdict(
        pearson_ehill_list_dev = [],
        pearson_ehill_list_test = [],
        pearson_list_perturbed_dev = [],
        pearson_list_perturbed_test = [],

        spearman_ehill_list_dev = [],
        spearman_ehill_list_test = [],
        spearman_list_perturbed_dev = [],
        spearman_list_perturbed_test = [],

        rmse_list_dev_ehill = [],
        rmse_list_test_ehill = [],
        rmse_list_perturbed_dev = [],
        rmse_list_perturbed_test = [],

        precisionk_list_perturbed_dev = [],
        precisionk_list_perturbed_test = []
    )

    model_training(args, model, data, hill_data, metrics_summary)
    report_final_results(metrics_summary)
    end_time = datetime.now()
    print(end_time - start_time)
