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
from collections import defaultdict


# check cuda
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print("Use GPU: %s" % torch.cuda.is_available())

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

def validation_epoch_end(epoch_loss_ehill, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary):
    print('Dev ehill loss:')
    print(epoch_loss_ehill / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'Dev ehill loss': epoch_loss_ehill/steps_per_epoch}, step=epoch)

    rmse_ehill = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_dev_ehill'].append(rmse_ehill)
    print('RMSE ehill: %.4f' % rmse_ehill)
    if USE_WANDB:
        wandb.log({'Dev ehill RMSE': rmse_ehill}, step=epoch)

    pearson_ehill, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_ehill_list_dev'].append(pearson_ehill)
    print('Pearson_ehill\'s correlation: %.4f' % pearson_ehill)
    if USE_WANDB:
        wandb.log({'Dev Pearson_ehill': pearson_ehill}, step = epoch)

    spearman_ehill, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_ehill_list_dev'].append(spearman_ehill)
    print('Spearman_ehill\'s correlation: %.4f' % spearman_ehill)
    if USE_WANDB:
        wandb.log({'Dev Spearman_ehill': spearman_ehill}, step = epoch)

def test_epoch_end(epoch_loss_ehill, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary):

    print('Test ehill loss:')
    print(epoch_loss_ehill / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'Test ehill Loss': epoch_loss_ehill / steps_per_epoch}, step = epoch)

    rmse_ehill = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_test_ehill'].append(rmse_ehill)
    print('RMSE ehill: %.4f' % rmse_ehill)
    if USE_WANDB:
        wandb.log({'Test RMSE ehill': rmse_ehill} , step = epoch)

    pearson_ehill, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_ehill_list_test'].append(pearson_ehill)
    print('Pearson_ehill\'s correlation: %.4f' % pearson_ehill)
    if USE_WANDB:
        wandb.log({'Test Pearson_ehill': pearson_ehill}, step = epoch)

    spearman_ehill, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_ehill_list_test'].append(spearman_ehill)
    print('Spearman_ehill\'s correlation: %.4f' % spearman_ehill)
    if USE_WANDB:
        wandb.log({'Test Spearman_ehill': spearman_ehill}, step = epoch)

def report_final_results(metrics_summary):
    best_dev_epoch = np.argmax(metrics_summary['spearman_ehill_list_dev'])
    print("Epoch %d got best Pearson's correlation of ehill on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['pearson_ehill_list_dev'][best_dev_epoch]))
    print("Epoch %d got Spearman's correlation of ehill on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['spearman_ehill_list_dev'][best_dev_epoch]))
    print("Epoch %d got RMSE of ehill on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['rmse_list_dev_ehill'][best_dev_epoch]))

    print("Epoch %d got Pearson's correlation of ehill on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['pearson_ehill_list_test'][best_dev_epoch]))
    print("Epoch %d got Spearman's correlation of ehill on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['spearman_ehill_list_test'][best_dev_epoch]))
    print("Epoch %d got RMSE of ehill on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['rmse_list_test_ehill'][best_dev_epoch]))

    best_test_epoch = np.argmax(metrics_summary['spearman_ehill_list_test'])
    print("Epoch %d got best Pearson's correlation of ehill on test set: %.4f" % (best_test_epoch + 1, metrics_summary['pearson_ehill_list_test'][best_test_epoch]))
    print("Epoch %d got Spearman's correlation of ehill on test set: %.4f" % (best_test_epoch + 1, metrics_summary['spearman_ehill_list_test'][best_test_epoch]))
    print("Epoch %d got RMSE of ehill on test set: %.4f" % (best_test_epoch + 1, metrics_summary['rmse_list_test_ehill'][best_test_epoch]))

def model_training(args, model,  hill_data, metrics_summary):

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    
    for epoch in range(args.max_epoch):

        print("Iteration %d:" % (epoch+1))
       

        model.eval()

        epoch_loss_ehill = 0
        lb_np = np.empty([0,])
        predict_np = np.empty([0,])
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(hill_data.test_dataloader()):

                ### add each peace of data to GPU to save the memory usage
                drug = ft['drug']
                mask = ft['mask'].to(device)
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(drug, hill_data.gene.to(device), mask, cell_feature, pert_idose,
                                job_id='pretraining', epoch = epoch)
               
                
                predict_np = np.concatenate((predict_np, predict.cpu().numpy().reshape(-1)), axis=0)

        
            sorted_test_input = pd.read_csv(args.hill_test_file).sort_values(['pert_id', 'cell_id' ,'pert_idose'])
            genes_ref_file = pd.read_csv(args.hill_dev_file)
            genes_cols =  genes_ref_file.columns[5:]
            assert sorted_test_input.shape[0] == predict_np.shape[0]
            predict_df = pd.DataFrame(predict_np, index = sorted_test_input.index, columns = genes_cols)
            
            result_df = pd.concat([sorted_test_input.iloc[:, :4], predict_df], axis = 1)


            print("=====================================write out data=====================================")
           
            result_df.loc[[x for x in range(len(result_df))],:].to_csv(args.predicted_result_for_testset, index = False)
            print('finish writing!! ')
       
    

if __name__ == '__main__':

    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='MultiDCP Ehill pretraining')
    parser.add_argument('--device',type=int, default=7)
    #parser.add_argument('--drug_file',type = str, default = 'MultiDCP_data/data/all_drugs_l1000.csv')
    parser.add_argument('--drug_file',type = str, default = 'MultiDCP_data/data/53_drugs.csv')
    parser.add_argument('--gene_file', type = str, default ='MultiDCP_data/data/gene_vector.csv')
    parser.add_argument('--dropout', type = float, default = 0.1)
    parser.add_argument('--hill_train_file', type = str,default = 'MultiDCP_data/data/ehill_data/high_confident_data_train.csv')
    parser.add_argument('--hill_dev_file', type = str,default = 'MultiDCP_data/data/ehill_data/high_confident_data_dev.csv')
    parser.add_argument('--hill_test_file', type = str,default =  '/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/ehill_data/sampled_hc2_test_data.csv')
    #parser.add_argument('--hill_test_file', type = str,default =  'MultiDCP_data/data/ehill_data/high_confident_data_dev.csv')
   
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--max_epoch', type=int, default = 1)
    parser.add_argument('--all_cells',type = str, default = 'MultiDCP_data/data/ehill_data/pretrain_cell_list_ehill.p')
    #parser.add_argument('--cell_ge_file', type = str,help='the file which used to map cell line to gene expression file', default = 'MultiDCP_data/data/adjusted_ccle_tcga_ad_tpm_log2.csv')
    parser.add_argument('--cell_ge_file', type = str,help='the file which used to map cell line to gene expression file', default = '/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/h2c_ccle_tcga_ad_batch_removal.csv')
    parser.add_argument('--linear_encoder_flag', dest = 'linear_encoder_flag', action='store_true', default=False,
                        help = 'whether the cell embedding layer only have linear layers')
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--trained_model_path',type = str,default='/raid/home/yoyowu/MultiDCP/saved_models/1013_ehill_rand2.pt')
    parser.add_argument('--predicted_result_for_testset',type=str, default='/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/predictions/0822test.csv')
    
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    all_cells = list(pickle.load(open(args.all_cells, 'rb')))
    DATA_FILTER = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
        "cell_id": all_cells,
        "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

    hill_data = datareader.EhillDataLoader(data_filter=None, device=device, args=args )
    # data = datareader.PerturbedDataLoader(DATA_FILTER, device, args)
    hill_data.setup()
    # data.setup()
   #print('#Train hill data: %d' % len(hill_data.train_data))
    #print('#Dev hill data: %d' % len(hill_data.dev_data))
    print('#Test hill data: %d' % len(hill_data.test_data))
    # print('#Train perturbed data: %d' % len(data.train_data))
    # print('#Dev perturbed data: %d' % len(data.dev_data))
    # print('#Test perturbed data: %d' % len(data.test_data))

    # parameters initialization
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
                                'pert_idose_input_dim': len(DATA_FILTER['pert_idose']),
                                'dropout': args.dropout, 
                                'linear_encoder_flag': args.linear_encoder_flag})

    # model creation
    print('--------------with linear encoder: {0!r}--------------'.format(args.linear_encoder_flag))
    model = multidcp.MultiDCPEhillPretraining(device=device, model_param_registry=model_param_registry)
    model.init_weights(pretrained = None)
    if args.trained_model_path is not None:
        model.load_state_dict(torch.load(args.trained_model_path))
        print(f'successfully loaded the model from{args.trained_model_path}')
    model.to(device)
    model = model.double()
    USE_WANDB = False
    PRECISION_DEGREE = [10, 20, 50, 100]
    if USE_WANDB:
        wandb.init(project="MultiDCP_AE_ehill",config=args)
        wandb.watch(model, log="all")
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        

    metrics_summary = defaultdict(
        pearson_ehill_list_dev = [],
        pearson_ehill_list_test = [],
        spearman_ehill_list_dev = [],
        spearman_ehill_list_test = [],
        rmse_list_dev_ehill = [],
        rmse_list_test_ehill = []
    )
    model_training(args, model,  hill_data, metrics_summary)
    #report_final_results(metrics_summary)
    end_time = datetime.now()
    print(end_time - start_time)
