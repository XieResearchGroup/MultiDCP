'''
probably will not use it forever
pretrain multidcp with auc and pic50 data
'''

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
from allrank.models.losses import approxNDCGLoss
import pickle

USE_wandb = True
if USE_wandb:
    wandb.init(project="MultiDCP")
else:
    os.environ["WANDB_MODE"] = "dryrun"

start_time = datetime.now()

parser = argparse.ArgumentParser(description='MultiDCP PreTraining')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--dropout')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size')
parser.add_argument('--max_epoch')
parser.add_argument('--all_cells')
parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file')
parser.add_argument('--linear_only', dest = 'linear_only', action='store_true', default=False,
                    help = 'whether the cell embedding layer only have linear layers')

args = parser.parse_args()

drug_file = args.drug_file
gene_file = args.gene_file
dropout = float(args.dropout)
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
linear_only = args.linear_only
print('--------------linear: {0!r}--------------'.format(linear_only))

all_cells = list(pickle.load(open(args.all_cells, 'rb')))
cell_ge_file = args.cell_ge_file

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 32
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
# precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse' #'point_wise_mse' # 'list_wise_ndcg'
intitializer = torch.nn.init.kaiming_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          #"cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "cell_id": all_cells,
          "pert_idose": ["10.0 um"]}

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

data = datareader.PerturbedDataLoader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                             gene_expression_file_test, filter, device, cell_ge_file, batch_size = batch_size)
data.setup()
print('#Train: %d' % len(data.train_data.feature['drug']))
print('#Dev: %d' % len(data.dev_data.feature['drug']))
print('#Test: %d' % len(data.test_data.feature['drug']))

# model creation
model = multidcp.MultiDCPPretraining(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=978,
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)
model.to(device)
model = model.double()
if USE_wandb:
     wandb.watch(model, log="all")

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
best_dev_loss = float("inf")
best_dev_pearson_pic50 = float("-inf")
best_dev_pearson_auc = float("-inf")
pearson_pic50_list_dev = []
pearson_auc_list_dev = []
pearson_pic50_list_test = []
pearson_auc_list_test = []
spearman_pic50_list_dev = []
spearman_auc_list_dev = []
spearman_pic50_list_test = []
spearman_auc_list_test = []
rmse_list_dev_pic50 = []
rmse_list_dev_auc = []
rmse_list_test_pic50 = []
rmse_list_test_auc = []
pearson_raw_list = []
for epoch in range(max_epoch):
    print("Iteration %d:" % (epoch+1))
    model.train()
    epoch_loss_pic50 = 0
    epoch_loss_auc = 0
    for i, batch in enumerate(data.train_dataloader()):
        ft, lb = batch
        drug = ft['drug']
        mask = ft['mask']
        if data.use_pert_type:
            pert_type = ft['pert_type']
        else:
            pert_type = None
        if data.use_cell_id:
            cell_id = ft['cell_id']
        else:
            cell_id = None
        if data.use_pert_idose:
            pert_idose = ft['pert_idose']
        else:
            pert_idose = None
        optimizer.zero_grad()
        predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose)
        #loss = approxNDCGLoss(predict, lb, padded_value_indicator=None)
        loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()
        loss_pic50 = model.loss(lb[:,0], predict[:,0])
        loss_auc = model.loss(lb[:,1], predict[:,1])
        epoch_loss_pic50 += loss_pic50.item()
        epoch_loss_auc += loss_auc.item()
    print('Train pic50 loss:')
    print(epoch_loss_pic50/(i+1))
    wandb.log({'Train pic50 loss': epoch_loss_pic50/(i+1)}, step = epoch)
    print('Train auc loss:')
    print(epoch_loss_auc/(i+1))
    wandb.log({'Train auc loss': epoch_loss_auc/(i+1)}, step = epoch)

    model.eval()

    epoch_loss_pic50 = 0
    epoch_loss_auc = 0
    lb_np = np.empty([0, 2])
    predict_np = np.empty([0, 2])
    with torch.no_grad():
        for i, batch in enumerate(data.val_dataloader()):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose)
            loss_pic50 = model.loss(lb[:,0], predict[:,0])
            loss_auc = model.loss(lb[:,1], predict[:,1])
            epoch_loss_pic50 += loss_pic50.item()
            epoch_loss_auc += loss_auc.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

        print('Dev pic50 loss:')
        print(epoch_loss_pic50 / (i + 1))
        wandb.log({'Dev pic50 loss': epoch_loss_pic50/(i+1)}, step=epoch)
        print('Dev auc loss:')
        print(epoch_loss_auc / (i + 1))
        wandb.log({'Dev auc loss': epoch_loss_auc/(i+1)}, step=epoch)

        rmse_pic50 = metric.rmse(lb_np[:,0], predict_np[:,0])
        rmse_list_dev_pic50.append(rmse_pic50)
        print('RMSE pic50: %.4f' % rmse_pic50)

        rmse_auc = metric.rmse(lb_np[:,1], predict_np[:,1])
        rmse_list_dev_auc.append(rmse_auc)
        print('RMSE auc: %.4f' % rmse_auc)

        wandb.log({'Dev pic50 RMSE': rmse_pic50}, step=epoch)
        wandb.log({'Dev auc RMSE': rmse_auc}, step=epoch)

        pearson_pic50, _ = metric.correlation(lb_np[:,0], predict_np[:,0], 'pearson')
        pearson_pic50_list_dev.append(pearson_pic50)
        print('Pearson_pic50\'s correlation: %.4f' % pearson_pic50)
        wandb.log({'Dev Pearson_pic50': pearson_pic50}, step = epoch)

        pearson_auc, _ = metric.correlation(lb_np[:,1], predict_np[:,1], 'pearson')
        pearson_auc_list_dev.append(pearson_auc)
        print('Pearson_auc\'s correlation: %.4f' % pearson_auc)
        wandb.log({'Dev Pearson_auc': pearson_auc}, step = epoch)

        spearman_pic50, _ = metric.correlation(lb_np[:,0], predict_np[:,0], 'spearman')
        spearman_pic50_list_dev.append(spearman_pic50)
        print('Spearman_pic50\'s correlation: %.4f' % spearman_pic50)
        wandb.log({'Dev Spearman_pic50': spearman_pic50}, step = epoch)

        spearman_auc, _ = metric.correlation(lb_np[:,1], predict_np[:,1], 'spearman')
        spearman_auc_list_dev.append(spearman_auc)
        print('Spearman_auc\'s correlation: %.4f' % spearman_auc)
        wandb.log({'Dev Spearman_auc': spearman_auc}, step = epoch)

        if best_dev_pearson_pic50 < pearson_pic50:
            best_dev_pearson_pic50 = pearson_pic50
            save(model.sub_multidcp.state_dict(), 'best_mode_storage_')
            print('==========================Best mode saved =====================')
        if best_dev_pearson_auc < pearson_auc:
            best_dev_pearson_auc = pearson_auc

    epoch_loss_pic50 = 0
    epoch_loss_auc = 0
    lb_np = np.empty([0, 2])
    predict_np = np.empty([0, 2])
    with torch.no_grad():
        for i, batch in enumerate(data.test_dataloader()):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose)
            loss_pic50 = model.loss(lb[:,0], predict[:,0])
            loss_auc = model.loss(lb[:,1], predict[:,1])
            epoch_loss_pic50 += loss_pic50.item()
            epoch_loss_auc += loss_auc.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

        print('Test pic50 loss:')
        print(epoch_loss_pic50 / (i + 1))
        wandb.log({'Test pic50 Loss': epoch_loss_pic50 / (i + 1)}, step = epoch)

        print('Test auc loss:')
        print(epoch_loss_auc / (i + 1))
        wandb.log({'Test auc Loss': epoch_loss_auc / (i + 1)}, step = epoch)

        rmse_pic50 = metric.rmse(lb_np, predict_np)
        rmse_list_test_pic50.append(rmse_pic50)
        print('RMSE pic50: %.4f' % rmse_pic50)
        wandb.log({'Test RMSE pic50': rmse_pic50} , step = epoch)

        rmse_auc = metric.rmse(lb_np, predict_np)
        rmse_list_test_auc.append(rmse_auc)
        print('RMSE auc: %.4f' % rmse_auc)
        wandb.log({'Test RMSE auc': rmse_auc} , step = epoch)

        pearson_pic50, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_pic50_list_test.append(pearson_pic50)
        print('Pearson_pic50\'s correlation: %.4f' % pearson_pic50)
        wandb.log({'Test Pearson_pic50': pearson_pic50}, step = epoch)

        pearson_auc, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_auc_list_test.append(pearson_auc)
        print('Pearson_auc\'s correlation: %.4f' % pearson_auc)
        wandb.log({'Test Pearson_auc': pearson_auc}, step = epoch)
        
        spearman_pic50, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_pic50_list_test.append(spearman_pic50)
        print('Spearman_pic50\'s correlation: %.4f' % spearman_pic50)
        wandb.log({'Test Spearman_pic50': spearman_pic50}, step = epoch)

        spearman_auc, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_auc_list_test.append(spearman_auc)
        print('Spearman_auc\'s correlation: %.4f' % spearman_auc)
        wandb.log({'Test Spearman_auc': spearman_auc}, step = epoch)

best_dev_epoch = np.argmax(spearman_auc_list_dev)
print("Epoch %d got best Pearson's correlation of pic50 on dev set: %.4f" % (best_dev_epoch + 1, pearson_pic50_list_dev[best_dev_epoch]))
print("Epoch %d got Spearman's correlation of pic50 on dev set: %.4f" % (best_dev_epoch + 1, spearman_pic50_list_dev[best_dev_epoch]))
print("Epoch %d got RMSE of pic50 on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_dev_pic50[best_dev_epoch]))

print("Epoch %d got best Pearson's correlation of auc on dev set: %.4f" % (best_dev_epoch + 1, pearson_auc_list_dev[best_dev_epoch]))
print("Epoch %d got Spearman's correlation of auc on dev set: %.4f" % (best_dev_epoch + 1, spearman_auc_list_dev[best_dev_epoch]))
print("Epoch %d got RMSE of auc on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_dev_auc[best_dev_epoch]))

print("Epoch %d got Pearson's correlation of pic50 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_pic50_list_test[best_dev_epoch]))
print("Epoch %d got Spearman's correlation of pic50 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_pic50_list_test[best_dev_epoch]))
print("Epoch %d got RMSE of pic50 on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_test_pic50[best_dev_epoch]))

print("Epoch %d got Pearson's correlation of auc on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_auc_list_test[best_dev_epoch]))
print("Epoch %d got Spearman's correlation of auc on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_auc_list_test[best_dev_epoch]))
print("Epoch %d got RMSE of auc on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_test_auc[best_dev_epoch]))

best_test_epoch = np.argmax(spearman_auc_list_test)
print("Epoch %d got best Pearson's correlation of pic50 on test set: %.4f" % (best_test_epoch + 1, pearson_pic50_list_test[best_test_epoch]))
print("Epoch %d got Spearman's correlation of pic50 on test set: %.4f" % (best_test_epoch + 1, spearman_pic50_list_test[best_test_epoch]))
print("Epoch %d got RMSE of pic50 on test set: %.4f" % (best_test_epoch + 1, rmse_list_test_pic50[best_test_epoch]))

print("Epoch %d got best Pearson's correlation of auc on test set: %.4f" % (best_test_epoch + 1, pearson_auc_list_test[best_test_epoch]))
print("Epoch %d got Spearman's correlation of auc on test set: %.4f" % (best_test_epoch + 1, spearman_auc_list_test[best_test_epoch]))
print("Epoch %d got RMSE of auc on test set: %.4f" % (best_test_epoch + 1, rmse_list_test_auc[best_test_epoch]))

end_time = datetime.now()
print(end_time - start_time)
