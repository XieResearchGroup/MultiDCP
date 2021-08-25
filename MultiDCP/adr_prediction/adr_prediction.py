'''
Train different models on the adr prediction dataset, this script is used for multi-label strategies
SVM, RF, Logistic
'''
import argparse
from adr_data_builder import ADRDataBuilder, PerturbedDGXDataBuilder, XYPreparer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import numpy as np
import pdb
from torch import device, cuda
from model_builder import DeepModel
import torch
import os
import wandb

# check cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU if it is deep model: %s" % torch.cuda.is_available())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--adr_file', help = 'the directory of file that have side effect data, *_PTs.csv')
    parser.add_argument('--gx_file', help = 'the directory of file that have the gene expression file, pred or truth, *DGX.csv')
    parser.add_argument('--cs_file', help = 'the directory of file that have MACC chemical strucutre information, MACC_bitmatrix')

    parser.add_argument('--pred_data', dest = 'pred_flag', action='store_true', default=False,
                    help = 'whether the gx data is perdicted data')
    parser.add_argument('--fold', help = 'the fold to be test in this script', type=int)

    args = parser.parse_args()
    adr_file = args.adr_file
    gx_file = args.gx_file
    pred_flag = args.pred_flag
    cs_file = args.cs_file
    fold = args.fold if args.fold < 5 else None

    adr_builder = ADRDataBuilder(file_name = adr_file)
    adr_drugs_list = adr_builder.get_drug_list()
    pertgx_builder = PerturbedDGXDataBuilder(gx_file_name=gx_file, drug_cs_dir=cs_file, pert_list = adr_drugs_list, pred_flag=pred_flag, cs_part = False)

    multi_label_models = [RandomForestClassifier, ExtraTreesClassifier, RidgeClassifierCV, MLPClassifier]
    
    X = pertgx_builder.get_gx_only()
    X_pertid_list = pertgx_builder.get_pert_id_list()
    
    Y = adr_builder.prepare_adr_df_basedon_perts(X_pertid_list)

    data_preparer = XYPreparer(X, Y, X_pertid_list, 42)

    cur_model_name = 'ExtraTrees'

    # set up logging system
    logging.basicConfig(filename='adr_prediction_log_' + cur_model_name.lower() + '_3_800_cs_more_adr_' + gx_file.rsplit('/',1)[1].rsplit('.',1)[0].lower(), 
                    level=logging.DEBUG,
                    format='%(asctime)-15s %(name)s %(levelname)s %(message)s')
    logger = logging.getLogger(name='ADR_Prediction')

    logger.debug("There are {0!r} drugs".format(len(set(X_pertid_list))))
    accuracy_micro_ls = []
    prauc_micro_ls = []
    accuracy_macro_ls = []
    prauc_macro_ls = []
    logger.debug("start 5-fold CV")
    for split_num, (train_index, test_index) in enumerate(data_preparer.leave_new_drug_out_split()):
        if fold and split_num != fold:
            continue
        logger.debug("the {0!r} split".format(split_num)) 
        Y_truth_with_zero = Y.values[test_index, :]
        Y_truth_train_with_zero = Y.values[train_index, :]
        nozero_filter = (Y_truth_with_zero.sum(axis = 0) > 0) & (Y_truth_train_with_zero.sum(axis = 0) > 0)
        logger.debug("nozero cols is {0!r}".format(sum(nozero_filter)))
        logger.debug("train data length {0!r}, test data length {1!r}".format(len(train_index), len(test_index)))
        # cur_model = RandomForestClassifier(n_estimators = 100, verbose = 2, n_jobs = 20)
        # cur_model = MLPClassifier(hidden_layer_sizes=200, verbose = 1)
        cur_model = ExtraTreesClassifier(n_estimators = 100, verbose = 2, n_jobs = 40)
        # cur_model = DeepModel(X_dim = X.shape[1], Y_dim = sum(nozero_filter), layout = [800,800], device = device, split_num=split_num, logger = logger)
        Y_truth = Y_truth_with_zero[:, nozero_filter]
        if isinstance(cur_model, DeepModel):
            best_metric = -float('inf')
            accuracy_micro_ls.append(float('inf'))
            accuracy_macro_ls.append(float('inf'))
            prauc_micro_ls.append(float('inf'))
            prauc_macro_ls.append(float('inf'))
            n_epochs = 500
            for i in range(n_epochs):
                loss = cur_model.fit(X.values[train_index,:], Y.values[train_index,:][:, nozero_filter])
                aucroc_micro, aucroc_macro, micro_prauc, macro_prauc = cur_model.score(X.values[test_index,:], Y.values[test_index,:][:, nozero_filter])
                if aucroc_macro > best_metric:
                    best_metric = aucroc_macro
                    accuracy_micro_ls[-1] = aucroc_micro
                    prauc_micro_ls[-1] = micro_prauc
                    accuracy_macro_ls[-1] = aucroc_macro
                    prauc_macro_ls[-1] = macro_prauc
            logger.debug("model prediction micro rocauc is {0:.4f}, macro rocauc is {1:.4f}".format(accuracy_micro_ls[-1], accuracy_macro_ls[-1]))
            logger.debug("model prediction micro prauc is {0:.4f}, macro prauc is {1:.4f}".format(prauc_micro_ls[-1], prauc_macro_ls[-1]))
        else:
            cur_model.fit(X.values[train_index,:], Y.values[train_index,:][:, nozero_filter])
            Y_pred = cur_model.predict_proba(X.values[test_index,:])
            logger.debug("Predict successfully")
            if isinstance(Y_pred, list):
                Y_pred = np.transpose([pred[:, 1] for pred in Y_pred])
            aucroc_micro = roc_auc_score(Y_truth.reshape(-1), Y_pred.reshape(-1))
            prauc_micro = average_precision_score(Y_truth.reshape(-1), Y_pred.reshape(-1))
            accuracy_micro_ls.append(aucroc_micro)
            prauc_micro_ls.append(prauc_micro)
            aucroc_macro = roc_auc_score(Y_truth, Y_pred, average = 'weighted')
            prauc_macro = average_precision_score(Y_truth, Y_pred, average = 'weighted')
            accuracy_macro_ls.append(aucroc_macro)
            prauc_macro_ls.append(prauc_macro)
            logger.debug("model prediction micro rocauc is {0:.4f}, macro rocauc is {1:.4f}".format(aucroc_micro, aucroc_macro))
            logger.debug("model prediction micro prauc is {0:.4f}, macro prauc is {1:.4f}".format(prauc_micro, prauc_macro))
    avg_aucroc_micro = np.mean(accuracy_micro_ls)
    avg_prauc_micro = np.mean(prauc_micro_ls)
    avg_aucroc_macro = np.mean(accuracy_macro_ls)
    avg_prauc_macro = np.mean(prauc_macro_ls)

    logger.debug("model prediction mean micro rocauc is {0:.4f}, mean macro rocauc is {1:.4f}".format(avg_aucroc_micro, avg_aucroc_macro))
    logger.debug("model prediction mean micro prauc is {0:.4f}, mean macro prauc is {1:.4f}".format(avg_prauc_micro, avg_prauc_macro))
