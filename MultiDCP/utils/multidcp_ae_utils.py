from collections import defaultdict
import torch
import metric
import wandb
import numpy as np

def initialize_model_registry():
    model_param_registry = defaultdict(lambda: None)
    model_param_registry['drug_input_dim'] = {'atom': 62, 'bond': 6}
    model_param_registry['drug_emb_dim'] = 128
    model_param_registry['conv_size'] = [16, 16]
    model_param_registry['degree'] = [0, 1, 2, 3, 4, 5]
    model_param_registry['gene_emb_dim'] = 128
    model_param_registry['gene_input_dim'] = 128
    model_param_registry['cell_id_input_dim'] = 978
    model_param_registry['cell_id_emb_dim'] = 50
    model_param_registry['cell_decoder_dim'] = 978 # autoencoder label's dimension
    model_param_registry['pert_idose_emb_dim'] = 4
    model_param_registry['hid_dim'] = 128
    model_param_registry['num_gene'] = 978
    model_param_registry['loss_type'] = 'point_wise_mse' # 'point_wise_mse' # 'list_wise_ndcg' #'combine'
    model_param_registry['initializer'] = torch.nn.init.kaiming_uniform_
    return model_param_registry

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("============current learning rate is {0!r}".format(param_group['lr']))

def validation_epoch_end(epoch_loss, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary, job, USE_WANDB):
    print('{0} Dev loss:'.format(job))
    print(epoch_loss / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'{0} Dev loss'.format(job): epoch_loss/steps_per_epoch}, step=epoch)
    rmse = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_{0}_dev'.format(job)].append(rmse)
    print('{0} RMSE: {1}'.format(job, rmse))
    if USE_WANDB:
        wandb.log({'{0} Dev RMSE'.format(job): rmse}, step=epoch)
    pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_list_{0}_dev'.format(job)].append(pearson)
    print('{0} Pearson\'s correlation: {1}'.format(job, pearson))
    if USE_WANDB:
        wandb.log({'{0} Dev Pearson'.format(job): pearson}, step = epoch)
    spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_list_{0}_dev'.format(job)].append(spearman)
    print('{0} Spearman\'s correlation: {1}'.format(job, spearman))
    if USE_WANDB:
        wandb.log({'{0} Dev Spearman'.format(job): spearman}, step = epoch)
    ae_precision = []
    for k in [10, 20, 50, 100]:
        precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
        print("{0} Precision@{1} Positive: {2}" .format(job, k, precision_pos))
        print("{0} Precision@{1} Negative: {2}" .format(job, k, precision_neg))
        ae_precision.append([precision_pos, precision_neg])
    metrics_summary['precisionk_list_{0}_dev'.format(job)].append(ae_precision)

def test_epoch_end(epoch_loss, lb_np, predict_np, steps_per_epoch, epoch, metrics_summary, job, USE_WANDB):
    print('{0} Test loss:'.format(job))
    print(epoch_loss / steps_per_epoch)
    if USE_WANDB:
        wandb.log({'{0} Test Loss'.format(job): epoch_loss / steps_per_epoch}, step = epoch)
    rmse = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_{0}_test'.format(job)].append(rmse)
    print('{0} RMSE: {1}'.format(job, rmse))
    if USE_WANDB:
        wandb.log({'{0} Test RMSE'.format(job): rmse} , step = epoch)
    pearson, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_list_{0}_test'.format(job)].append(pearson)
    print('{0} Pearson\'s correlation: {1}'.format(job, pearson))
    if USE_WANDB:
        wandb.log({'{0} Test Pearson'.format(job): pearson}, step = epoch)
    spearman, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_list_{0}_test'.format(job)].append(spearman)
    print('{0} Spearman\'s correlation: {1}'.format(job, spearman))
    if USE_WANDB:
        wandb.log({'{0} Test Spearman'.format(job): spearman}, step = epoch)
    ae_precision_test = []
    for k in [10, 20, 50, 100]:
        precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k)
        print("{0} Precision@{1} Positive: {2}".format(job, k, precision_pos))
        print("{0} Precision@{1} Negative: {2}".format(job, k, precision_neg))
        ae_precision_test.append([precision_pos, precision_neg])
    metrics_summary['precisionk_list_{0}_test'.format(job)].append(ae_precision_test)

def report_final_results(metrics_summary, ae = False, perturbed = False):
    
    if ae:
        best_ae_dev_epoch = np.argmax(metrics_summary['pearson_list_ae_dev'])
        print("Epoch %d got best AE Pearson's correlation on dev set: %.4f" % (best_ae_dev_epoch + 1, metrics_summary['pearson_list_ae_dev'][best_ae_dev_epoch]))
        print("Epoch %d got AE Spearman's correlation on dev set: %.4f" % (best_ae_dev_epoch + 1, metrics_summary['spearman_list_ae_dev'][best_ae_dev_epoch]))
        print("Epoch %d got AE RMSE on dev set: %.4f" % (best_ae_dev_epoch + 1, metrics_summary['rmse_list_ae_dev'][best_ae_dev_epoch]))
        print("Epoch %d got AE P@100 POS and NEG on dev set: %.4f, %.4f" % (best_ae_dev_epoch + 1,
                                                                        metrics_summary['precisionk_list_ae_dev'][best_ae_dev_epoch][-1][0],
                                                                        metrics_summary['precisionk_list_ae_dev'][best_ae_dev_epoch][-1][1]))

        print("Epoch %d got AE Pearson's correlation on test set w.r.t dev set: %.4f" % (best_ae_dev_epoch + 1, metrics_summary['pearson_list_ae_test'][best_ae_dev_epoch]))
        print("Epoch %d got AE Spearman's correlation on test set w.r.t dev set: %.4f" % (best_ae_dev_epoch + 1, metrics_summary['spearman_list_ae_test'][best_ae_dev_epoch]))
        print("Epoch %d got AE RMSE on test set w.r.t dev set: %.4f" % (best_ae_dev_epoch + 1, metrics_summary['rmse_list_ae_test'][best_ae_dev_epoch]))
        print("Epoch %d got AE P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_ae_dev_epoch + 1,
                                                                        metrics_summary['precisionk_list_ae_test'][best_ae_dev_epoch][-1][0],
                                                                        metrics_summary['precisionk_list_ae_test'][best_ae_dev_epoch][-1][1]))

        best_ae_test_epoch = np.argmax(metrics_summary['pearson_list_ae_test'])
        print("Epoch %d got AE best Pearson's correlation on test set: %.4f" % (best_ae_test_epoch + 1, metrics_summary['pearson_list_ae_test'][best_ae_test_epoch]))
        print("Epoch %d got AE Spearman's correlation on test set: %.4f" % (best_ae_test_epoch + 1, metrics_summary['spearman_list_ae_test'][best_ae_test_epoch]))
        print("Epoch %d got AE RMSE on test set: %.4f" % (best_ae_test_epoch + 1, metrics_summary['rmse_list_ae_test'][best_ae_test_epoch]))
        print("Epoch %d got AE P@100 POS and NEG on test set: %.4f, %.4f" % (best_ae_test_epoch + 1,
                                                                        metrics_summary['precisionk_list_ae_test'][best_ae_test_epoch][-1][0],
                                                                        metrics_summary['precisionk_list_ae_test'][best_ae_test_epoch][-1][1]))

    if perturbed:
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
