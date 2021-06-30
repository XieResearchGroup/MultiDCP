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

USE_wandb = True
if USE_wandb:
    wandb.init(project="MultiDCP_AE_loss")
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
parser.add_argument('--unfreeze_steps', help='The epochs at which each layer is unfrozen, like <<1,2,3,4>>')
parser.add_argument('--ae_input_file')
parser.add_argument('--ae_label_file')
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
unfreeze_steps = args.unfreeze_steps.split(',')
assert len(unfreeze_steps) == 4, "number of unfreeze steps should be 4"
unfreeze_pattern = [False, False, False, False]
ae_input_file = args.ae_input_file
ae_label_file = args.ae_label_file
cell_ge_file = args.cell_ge_file
linear_only = args.linear_only
print('--------------linear: {0!r}--------------'.format(linear_only))

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 32
cell_decoder_dim = 978 # autoencoder label's dimension
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse' #'point_wise_mse' # 'list_wise_ndcg'
intitializer = torch.nn.init.kaiming_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          #"cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "cell_id": ['A549', 'MCF7', 'HCC515', 'HEPG2', 'HS578T', 'PC3', 'SKBR3', 'MDAMB231', 'JURKAT', 'A375', 'BT20', 'HELA', 'HT29', 'HA1E', 'YAPC'],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

ae_data = datareader.AEDataReader(ae_input_file, ae_label_file, device)
data = datareader.PerturbedDataLoader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                             gene_expression_file_test, filter, device, cell_ge_file, batch_size = batch_size)
data.setup()
print('#Train: %d' % len(data.train_data.feature['drug']))
print('#Dev: %d' % len(data.dev_data.feature['drug']))
print('#Test: %d' % len(data.test_data.feature['drug']))

# model creation
model = multidcp.MultiDCP_AE(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=978,
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, cell_decoder_dim = cell_decoder_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)
model.init_weights(pretrained = True)
model.to(device)
model = model.double()
if USE_wandb:
     wandb.watch(model, log="all")

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: step_lr([int(x) for x in unfreeze_steps], x)])
best_dev_loss = float("inf")
best_dev_pearson = float("-inf")
pearson_list_ae_dev = []
pearson_list_ae_test = []
pearson_list_perturbed_dev = []
pearson_list_perturbed_test = []
spearman_list_ae_dev = []
spearman_list_ae_test = []
spearman_list_perturbed_dev = []
spearman_list_perturbed_test = []
rmse_list_ae_dev = []
rmse_list_ae_test = []
rmse_list_perturbed_dev = []
rmse_list_perturbed_test = []
precisionk_list_ae_dev = []
precisionk_list_ae_test = []
precisionk_list_perturbed_dev = []
precisionk_list_perturbed_test = []
pearson_raw_list_ae = []
pearson_raw_list_perturbed = []
for epoch in range(max_epoch):

    scheduler.step()
    for param_group in optimizer.param_groups:
        print("============current learning rate is {0!r}".format(param_group['lr']))
    if str(epoch) in unfreeze_steps:
        number_layer_to_unfreeze = 3 - unfreeze_steps[::-1].index(str(epoch)) ## find the position of last occurance of number epoch
        for i in range(3-number_layer_to_unfreeze,4):
            unfreeze_pattern[i] = True
        model.gradual_unfreezing(unfreeze_pattern)

    print(unfreeze_pattern)
    print("Iteration %d:" % (epoch+1))
    model.train()

    epoch_loss = 0

    for i, (ft, lb, cell_type) in enumerate(data.train_dataloader()):
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
        predict, cell_hidden_ = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
                                      epoch = epoch, linear_only = linear_only)
        #loss = approxNDCGLoss(predict, lb, padded_value_indicator=None)
        loss = model.loss(lb, predict)
        loss_2 = apply_NodeHomophily(cell_hidden_, cell_type)
        loss_t = loss + 0.001 * loss_2
        loss.backward()
        optimizer.step()
        print(loss.item(), loss_2.item())
        if i == 1:
            print('__________________________input___________________________')
            print(cell_id)
            print('__________________________prediction___________________________')
            print(predict)
            print('__________________________hidden__________________________')
            print(cell_hidden_)
        epoch_loss += loss.item()
    print('Perturbed gene expression profile Train loss:')
    print(epoch_loss/(i+1))
    if USE_wandb:
        wandb.log({'Perturbed gene expression profile Train loss': epoch_loss/(i+1)}, step = epoch)

    model.eval()

    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(data.val_dataloader()):
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
            predict, _ = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
                               epoch = epoch, linear_only = linear_only)
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        print('Perturbed gene expression profile Dev loss:')
        print(epoch_loss / (i + 1))
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Dev loss': epoch_loss/(i+1)}, step=epoch)
        rmse = metric.rmse(lb_np, predict_np)
        torch.save(lb_np, 'lb_np_sec.pt')
        torch.save(predict_np, 'predict_np_sec.pt')
        rmse_list_perturbed_dev.append(rmse)
        print('Perturbed gene expression profile RMSE: %.4f' % rmse)
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Dev RMSE': rmse}, step=epoch)
        pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_list_perturbed_dev.append(pearson)
        print('Perturbed gene expression profile Pearson\'s correlation: %.4f' % pearson)
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Dev Pearson': pearson}, step = epoch)
        spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_list_perturbed_dev.append(spearman)
        print('Perturbed gene expression profile Spearman\'s correlation: %.4f' % spearman)
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Dev Spearman': spearman}, step = epoch)
        perturbed_precision = []
        for k in precision_degree:
            precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
            print("Perturbed gene expression profile Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Perturbed gene expression profile Precision@%d Negative: %.4f" % (k, precision_neg))
            # if E_wandb:
            # wandb.log({'Perturbed gene expression profile Dev Precision Positive@{0!r}'.format(k): precision_pos}, step = epoch)
            # if E_wandb:
            # wandb.log({'Perturbed gene expression profile Dev Precision Negative@{0!r}'.format(k): precision_neg}, step = epoch)
            perturbed_precision.append([precision_pos, precision_neg])
        precisionk_list_perturbed_dev.append(perturbed_precision)

        if best_dev_pearson < pearson:
            best_dev_pearson = pearson

    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, (ft, lb, _) in enumerate(data.test_dataloader()):
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
            predict, _ = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
                               epoch=epoch, linear_only=linear_only)
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        print('Perturbed gene expression profile Test loss:')
        print(epoch_loss / (i + 1))
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Test Loss': epoch_loss / (i + 1)}, step = epoch)
        rmse = metric.rmse(lb_np, predict_np)
        rmse_list_perturbed_test.append(rmse)
        print('Perturbed gene expression profile RMSE: %.4f' % rmse)
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Test RMSE': rmse} , step = epoch)
        pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
        pearson_list_perturbed_test.append(pearson)
        print('Perturbed gene expression profile Pearson\'s correlation: %.4f' % pearson)
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Test Pearson': pearson}, step = epoch)
        spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
        spearman_list_perturbed_test.append(spearman)
        print('Perturbed gene expression profile Spearman\'s correlation: %.4f' % spearman)
        if USE_wandb:
            wandb.log({'Perturbed gene expression profile Test Spearman': spearman}, step = epoch)
        perturbed_precision_test = []
        for k in precision_degree:
            precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
            print("Perturbed gene expression profile Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Perturbed gene expression profile Precision@%d Negative: %.4f" % (k, precision_neg))
            # if E_wandb:
            # wandb.log({'Perturbed gene expression profile Test Precision Positive@{0!r}'.format(k): precision_pos}, step=epoch)
            # if E_wandb:
            # wandb.log({'Perturbed gene expression profile Test Precision Negative@{0!r}'.format(k): precision_neg}, step=epoch)
            perturbed_precision_test.append([precision_pos, precision_neg])
        precisionk_list_perturbed_test.append(perturbed_precision_test)

best_dev_epoch = np.argmax(pearson_list_perturbed_dev)

print("Epoch %d got best Perturbed Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, pearson_list_perturbed_dev[best_dev_epoch]))
print("Epoch %d got Perturbed Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, spearman_list_perturbed_dev[best_dev_epoch]))
print("Epoch %d got Perturbed RMSE on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_perturbed_dev[best_dev_epoch]))
print("Epoch %d got Perturbed P@100 POS and NEG on dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_perturbed_dev[best_dev_epoch][-1][0],
                                                                  precisionk_list_perturbed_dev[best_dev_epoch][-1][1]))

print("Epoch %d got Perturbed Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_list_perturbed_test[best_dev_epoch]))
print("Epoch %d got Perturbed Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_list_perturbed_test[best_dev_epoch]))
print("Epoch %d got Perturbed RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_perturbed_test[best_dev_epoch]))
print("Epoch %d got Perturbed P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_perturbed_test[best_dev_epoch][-1][0],
                                                                  precisionk_list_perturbed_test[best_dev_epoch][-1][1]))

best_test_epoch = np.argmax(pearson_list_perturbed_test)

print("Epoch %d got Perturbed best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, pearson_list_perturbed_test[best_test_epoch]))
print("Epoch %d got Perturbed Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, spearman_list_perturbed_test[best_test_epoch]))
print("Epoch %d got Perturbed RMSE on test set: %.4f" % (best_test_epoch + 1, rmse_list_perturbed_test[best_test_epoch]))
print("Epoch %d got Perturbed P@100 POS and NEG on test set: %.4f, %.4f" % (best_test_epoch + 1,
                                                                  precisionk_list_perturbed_test[best_test_epoch][-1][0],
                                                                  precisionk_list_perturbed_test[best_test_epoch][-1][1]))

end_time = datetime.now()
print(end_time - start_time)