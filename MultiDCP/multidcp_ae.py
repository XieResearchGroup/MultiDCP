import os
import pandas as pd
import torch.nn as nn 
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
from sklearn import metrics
import wandb
import pdb
import pickle
from loss_utils import apply_NodeHomophily
from tqdm import tqdm
import random
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")


# check cuda


  
#print("Use GPU: %s" % torch.cuda.is_available())

def initialize_model_registry():
    model_param_registry = defaultdict(lambda: None)
    model_param_registry['drug_input_dim'] = {'atom': 62, 'bond': 6}
    model_param_registry['drug_emb_dim'] = 128
    model_param_registry['conv_size'] = [16, 16]
    model_param_registry['degree'] = [0, 1, 2, 3, 4, 5]
    model_param_registry['gene_emb_dim'] = 128
    model_param_registry['gene_input_dim'] = 128
    model_param_registry['cell_id_input_dim'] = 978
    model_param_registry['cell_decoder_dim'] = 978 # autoencoder label's dimension
    model_param_registry['pert_idose_emb_dim'] = 4
    model_param_registry['hid_dim'] = 128
    model_param_registry['num_gene'] = 978
    model_param_registry['loss_type'] = 'point_wise_mse' #'point_wise_mse' # 'list_wise_ndcg' #'combine'
    model_param_registry['initializer'] = torch.nn.init.kaiming_uniform_
    return model_param_registry

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("============current learning rate is {0!r}".format(param_group['lr']))

def validation_epoch_end(epoch_loss, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary, job):
    print('{0} Dev loss:'.format(job))
    print(epoch_loss / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'{0} Dev loss'.format(job): epoch_loss/steps_per_epoch}, step=epoch)
    lb_np = lb_np.flatten()
    predict_np = predict_np.flatten()
    
    aucroc = metrics.roc_auc_score(lb_np, predict_np)
    metrics_summary['aucroc_list_{0}_dev'.format(job)].append(aucroc)
    print('{0} AUCROC: {1}'.format(job, aucroc))
    if USE_WANDB:
        wandb.log({'{0} Dev AUCROC'.format(job): aucroc}, step=epoch)
    prec,reca,_ = metrics.precision_recall_curve(lb_np, predict_np)
    aucpr = metrics.auc(reca,prec)

    metrics_summary['aucpr_{0}_dev'.format(job)].append(aucpr)
    print('{0} AUCPR: {1}'.format(job, aucpr))
    if USE_WANDB:
        wandb.log({'{0} Dev AUCPR'.format(job): aucpr}, step = epoch)
    
def test_epoch_end(epoch_loss, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary, job):

    print('{0} Test loss:'.format(job))
    print(epoch_loss / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'{0} Test loss'.format(job): epoch_loss/steps_per_epoch}, step=epoch)

    lb_np = lb_np.flatten()
    predict_np = predict_np.flatten()
    
    aucroc = metrics.roc_auc_score(lb_np, predict_np)
    metrics_summary['aucroc_list_{0}_test'.format(job)].append(aucroc)
    print('{0} AUCROC: {1}'.format(job, aucroc))
    if USE_WANDB:
        wandb.log({'{0} Test AUCROC'.format(job): aucroc}, step=epoch)
    prec,reca,_ = metrics.precision_recall_curve(lb_np, predict_np)
    aucpr = metrics.auc(reca,prec)

    metrics_summary['aucpr_{0}_test'.format(job)].append(aucpr)
    print('{0} AUCPR: {1}'.format(job, aucpr))
    if USE_WANDB:
        wandb.log({'{0} Test AUCPR'.format(job): aucpr}, step = epoch)
    
  


def report_final_results(metrics_summary):
    best_dev_epoch = np.argmax(metrics_summary['aucroc_list_perturbed_dev'])
  
    print("Epoch %d got aucroc on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['aucroc_list_perturbed_dev'][best_dev_epoch]))
    print("Epoch %d got auprc on dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['aucpr_perturbed_dev'][best_dev_epoch]))
   

    print("Epoch %d got Perturbed aucroc on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['aucroc_list_perturbed_test'][best_dev_epoch]))
    print("Epoch %d got Perturbed auprc on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, metrics_summary['aucpr_perturbed_test'][best_dev_epoch]))

    best_test_epoch = np.argmax(metrics_summary['aucroc_list_perturbed_test'])


    print("Epoch %d got Perturbed best aucroc on test set: %.4f" % (best_test_epoch + 1, metrics_summary['aucroc_list_perturbed_test'][best_test_epoch]))
    print("Epoch %d got Perturbed auprc on test set: %.4f" % (best_test_epoch + 1, metrics_summary['aucpr_perturbed_test'][best_test_epoch]))


def model_training(args, model, data, ae_data, metrics_summary):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dev_aucroc = float("-inf")

    for epoch in range(args.max_epoch):
    
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

        epoch_loss = 0
        for i, (ft, lb) in enumerate(data.train_dataloader()):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            optimizer.zero_grad()
            predict, cell_hidden_ = model(input_cell_gex=cell_feature, input_drug = drug, 
                                        input_gene = data.gene, mask = mask,
                                        input_pert_idose = pert_idose, 
                                        job_id = 'perturbed', epoch = epoch)
            loss_t = nn.BCEWithLogitsLoss()( predict,lb)
            loss_t.backward()
            optimizer.step()
            if i == 1:
                print('__________________________pertubed input__________________________')
                print(cell_feature)
                print('__________________________pertubed hidden__________________________')
                print(cell_hidden_)
                print('__________________________pertubed predicts__________________________')
                print(cell_hidden_)
            epoch_loss += loss_t.item()
        print('Perturbed gene expression profile Train loss:')
        print(epoch_loss/(i+1))
        if USE_WANDB:
            wandb.log({'Perturbed gene expression profile Train loss': epoch_loss/(i+1)}, step = epoch)

        model.eval()
        epoch_loss = 0
        lb_np = np.empty([0, 978])
        predict_np = np.empty([0, 978])
        with torch.no_grad():
            for i, (ft, lb) in enumerate(data.val_dataloader()):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(input_cell_gex=cell_feature, input_drug = drug, 
                                input_gene = data.gene, mask = mask,
                                input_pert_idose = pert_idose, 
                                job_id = 'perturbed', epoch = epoch)
                loss = nn.BCEWithLogitsLoss()( predict,lb)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            validation_epoch_end(epoch_loss = epoch_loss, lb_np = lb_np, 
                                predict_np = predict_np, steps_per_epoch = i+1, 
                                epoch = epoch, metrics_summary = metrics_summary,
                                job = 'perturbed')

            if best_dev_aucroc < metrics_summary['aucroc_list_perturbed_dev'][-1] or epoch == 1:
                # data_save = True
                best_dev_aucroc = metrics_summary['aucroc_list_perturbed_dev'][-1]
                torch.save(model.multidcp.state_dict(), args.model_save_path)
    
        epoch_loss = 0
        lb_np_ls = []
        predict_np_ls = []
        hidden_np_ls = []
        with torch.no_grad():
            for i, (ft, lb) in enumerate(tqdm(data.test_dataloader())):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, cells_hidden_repr = model(input_cell_gex=cell_feature, input_drug = drug, 
                                                input_gene = data.gene, mask = mask,
                                                input_pert_idose = pert_idose, job_id = 'perturbed')
                loss =nn.BCEWithLogitsLoss()(predict,lb)
                epoch_loss += loss.item()
                lb_np_ls.append(lb.cpu().numpy()) 
                predict_np_ls.append(predict.cpu().numpy()) 
                hidden_np_ls.append(cells_hidden_repr.cpu().numpy()) 

            lb_np = np.concatenate(lb_np_ls, axis = 0)
            predict_np = np.concatenate(predict_np_ls, axis = 0)
            hidden_np = np.concatenate(hidden_np_ls, axis = 0)
           
            test_epoch_end(epoch_loss = epoch_loss, lb_np = lb_np, 
                                predict_np = predict_np, steps_per_epoch = i+1, 
                                epoch = epoch, metrics_summary = metrics_summary,
                                job = 'perturbed')


if __name__ == '__main__':
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='MultiDCP binary PreTraining')
    parser.add_argument('--expname', type= str, default='')
    parser.add_argument('--device',type=int, default=0)
    parser.add_argument('--drug_file',type = str, default="MultiDCP_data/data/all_drugs_l1000.csv")
    parser.add_argument('--lr',type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=343)
    parser.add_argument('--gene_file', type = str, default = "MultiDCP_data/data/gene_vector.csv")
    parser.add_argument('--train_file', type= str, default= "/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_train_2.csv")
    parser.add_argument('--dev_file', type=str, default ="/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_dev_2.csv" )
    parser.add_argument('--test_file', type=str, default="/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_test_2.csv")
    parser.add_argument('--batch_size', type = int, default=64)
    parser.add_argument('--ae_input_file',type=str, default= "MultiDCP_data/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4" )
    parser.add_argument('--ae_label_file', type=str, default="MultiDCP_data/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    parser.add_argument('--cell_ge_file',type=str,default="MultiDCP_data/data/adjusted_ccle_tcga_ad_tpm_log2.csv",
                        help='the file which used to map cell line to gene expression file', )
    parser.add_argument('--max_epoch', type = int, default=500)
    parser.add_argument('--predicted_result_for_testset',  type=str, default="MultiDCP_data/data/teacher_student/teach_stu_perturbedGX.csv")
    parser.add_argument('--hidden_repr_result_for_testset', type=str, default = "MultiDCP_data/data/teacher_student/teach_stu_perturbedGX_hidden.csv" ,
                        help = "the file directory to save the test data hidden representation dataframe")
    parser.add_argument('--all_cells', type=str, default= "MultiDCP_data/data/ccle_tcga_ad_cells.p")
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--linear_encoder_flag', dest = 'linear_encoder_flag', action='store_true', default=False,
                        help = 'whether the cell embedding layer only have linear layers')
    parser.add_argument('--model_save_path',type=str, default='')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
   
    all_cells = list(pickle.load(open(args.all_cells, 'rb')))
    DATA_FILTER = {"time": "24H",
            "pert_type": ["trt_cp"],
            "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

   
    ae_data = datareader.AEDataLoader(device, args)
    data = datareader.PerturbedDataLoader(DATA_FILTER,device, args)
    ae_data.setup()
    data.setup()
    print('#Train: %d' % len(data.train_data))
    print('#Dev: %d' % len(data.dev_data))
    print('#Test: %d' % len(data.test_data))
    print('#Train AE: %d' % len(ae_data.train_data))
    # print('#Dev AE: %d' % len(ae_data.dev_data))
    # print('#Test AE: %d' % len(ae_data.test_data))

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


    USE_WANDB = True
    PRECISION_DEGREE = [10, 20, 50, 100]
    if USE_WANDB:
        wandb.init(project="MultiDCP_binary",config=args)
        wandb.watch(model, log="all")
    else:
        os.environ["WANDB_MODE"] = "dryrun"

    # training
    metrics_summary = defaultdict(
        aucroc_list_perturbed_dev = [],
        aucroc_list_perturbed_test = [],
        aucpr_perturbed_dev = [],
        aucpr_perturbed_test = [])

    model_training(args, model, data, ae_data, metrics_summary)
    report_final_results(metrics_summary)
    end_time = datetime.now()
    print(end_time - start_time)
