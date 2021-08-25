import numpy as np
import random
import torch
import data_utils
import pandas as pd
import pdb
import warnings
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
warnings.filterwarnings("ignore")

seed = 343
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)

class AEDataDataset(Dataset):

    def __init__(self, input_file_name, label_file_name, device):
        super(AEDataDataset, self).__init__()
        self.device = device
        self.feature = torch.from_numpy(np.asarray(pd.read_csv(input_file_name, index_col=0).values, dtype=np.float64)).to(device)
        self.label = torch.from_numpy(np.asarray(pd.read_csv(label_file_name, index_col=0).values, dtype=np.float64)).to(device)
        self.cell_type_code = torch.Tensor([*range(len(self.feature))]).long()

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx], self.cell_type_code[idx]

class AEDataLoader(pl.LightningDataModule):

    def __init__(self, device, args):
        super(AEDataLoader, self).__init__()
        self.batch_size = args.batch_size
        self.train_data_file = args.ae_input_file + '_train.csv'
        self.dev_data_file = args.ae_input_file + '_dev.csv'
        self.test_data_file = args.ae_input_file + '_test.csv'
        self.train_label_file = args.ae_label_file + '_train.csv'
        self.dev_label_file = args.ae_label_file + '_dev.csv'
        self.test_label_file = args.ae_label_file + '_test.csv'
        self.device = device

    def prepare_data(self):
        '''
        Use this method to do things that might write to disk or that need to be \
            done only from a single GPU in distributed settings.
        how to download(), tokenize, the processed file need to be saved to disk to be accessed by other processes
        prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        '''
        pass

    def setup(self, stage = None):
        self.train_data = AEDataDataset(self.train_data_file, self.train_label_file, self.device)
        self.dev_data = AEDataDataset(self.dev_data_file, self.dev_label_file, self.device)
        self.test_data = AEDataDataset(self.test_data_file, self.test_label_file, self.device)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size = self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size)


class PerturbedDataset(Dataset):

    def __init__(self, drug_file, data_file, data_filter, device, cell_ge_file_name):
        super(PerturbedDataset, self).__init__()
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        feature, label, self.cell_type = data_utils.read_data(data_file, data_filter)
        self.feature, self.label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transform_to_tensor_per_dataset(feature, label, self.drug, self.device, cell_ge_file_name)

    def __len__(self):
        return self.feature['drug'].shape[0]

    def __getitem__(self, idx):
        output = dict()
        output['drug'] = self.feature['drug'][idx]
        if self.use_cell_id:
            output['cell_id'] = self.feature['cell_id'][idx]
        if self.use_pert_idose:
            output['pert_idose'] = self.feature['pert_idose'][idx]
        return output, self.label[idx], self.cell_type[idx]

class PerturbedDataLoader(pl.LightningDataModule):

    def __init__(self, data_filter, device, args):
        super(PerturbedDataLoader, self).__init__()
        self.batch_size = args.batch_size
        self.train_data_file = args.train_file
        self.dev_data_file = args.dev_file
        self.test_data_file = args.test_file
        self.drug_file = args.drug_file
        self.data_filter = data_filter
        self.device = device
        self.cell_ge_file_name = args.cell_ge_file
        self.gene = data_utils.read_gene(args.gene_file, self.device)
    
    def collate_fn(self, batch):
        features = {}
        features['drug'] = data_utils.convert_smile_to_feature([output['drug'] for output, _, _ in batch], self.device)
        features['mask'] = data_utils.create_mask_feature(features['drug'], self.device)
        for key in batch[0][0].keys():
            if key == 'drug':
                continue
            features[key] = torch.stack([output[key] for output, _, _ in batch], dim = 0)
        labels = torch.stack([label for _, label, _ in batch], dim = 0)
        cell_types = torch.Tensor([cell_type for _, _, cell_type in batch])
        return features, labels, torch.Tensor(cell_types).to(self.device)

    def prepare_data(self):
        '''
        Use this method to do things that might write to disk or that need to be \
            done only from a single GPU in distributed settings.
        how to download(), tokenize, the processed file need to be saved to disk to be accessed by other processes
        prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        '''
        pass

    def setup(self, stage = None):
        self.train_data = PerturbedDataset(self.drug_file, self.train_data_file,
                 self.data_filter, self.device, self.cell_ge_file_name)
        self.dev_data = PerturbedDataset(self.drug_file, self.dev_data_file,
                 self.data_filter, self.device, self.cell_ge_file_name)
        self.test_data = PerturbedDataset(self.drug_file, self.test_data_file,
                 self.data_filter, self.device, self.cell_ge_file_name)
        self.use_pert_type = self.train_data.use_pert_type
        self.use_cell_id = self.train_data.use_cell_id
        self.use_pert_idose = self.train_data.use_pert_idose
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size = self.batch_size, collate_fn = self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, collate_fn = self.collate_fn)

class EhillDataset(Dataset):

    def __init__(self, drug_file, data_file, data_filter, device, cell_ge_file_name):
        super(EhillDataset, self).__init__()
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        feature, label, self.cell_type = data_utils.read_data(data_file, data_filter)
        self.feature, self.label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transform_to_tensor_per_dataset(feature, label, self.drug, self.device, cell_ge_file_name)

    def __len__(self):
        return self.feature['drug'].shape[0]

    def __getitem__(self, idx):
        output = dict()
        output['drug'] = self.feature['drug'][idx]
        if self.use_cell_id:
            output['cell_id'] = self.feature['cell_id'][idx]
        if self.use_pert_idose:
            output['pert_idose'] = self.feature['pert_idose'][idx]
        return output, self.label[idx], self.cell_type[idx]

class EhillDataLoader(pl.LightningDataModule):

    def __init__(self, data_filter, device, args):
        super(EhillDataLoader, self).__init__()
        self.batch_size = args.batch_size
        self.train_data_file = args.hill_train_file
        self.dev_data_file = args.hill_dev_file
        self.test_data_file = args.hill_test_file
        self.drug_file = args.drug_file
        self.data_filter = data_filter
        self.device = device
        self.cell_ge_file_name = args.cell_ge_file
        self.gene = data_utils.read_gene(args.gene_file, self.device)
    
    def collate_fn(self, batch):
        features = {}
        features['drug'] = data_utils.convert_smile_to_feature([output['drug'] for output, _, _ in batch], self.device)
        features['mask'] = data_utils.create_mask_feature(features['drug'], self.device)
        for key in batch[0][0].keys():
            if key == 'drug':
                continue
            features[key] = torch.stack([output[key] for output, _, _ in batch], dim = 0)
        labels = torch.stack([label for _, label, _ in batch], dim = 0)
        cell_types = torch.Tensor([cell_type for _, _, cell_type in batch])
        return features, labels, torch.Tensor(cell_types).to(self.device)

    def prepare_data(self):
        '''
        Use this method to do things that might write to disk or that need to be \
            done only from a single GPU in distributed settings.
        how to download(), tokenize, the processed file need to be saved to disk to be accessed by other processes
        prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        '''
        pass

    def setup(self, stage = None):
        self.train_data = EhillDataset(self.drug_file, self.train_data_file,
                 self.data_filter, self.device, self.cell_ge_file_name)
        self.dev_data = EhillDataset(self.drug_file, self.dev_data_file,
                 self.data_filter, self.device, self.cell_ge_file_name)
        self.test_data = EhillDataset(self.drug_file, self.test_data_file,
                 self.data_filter, self.device, self.cell_ge_file_name)
        self.use_pert_type = self.train_data.use_pert_type
        self.use_cell_id = self.train_data.use_cell_id
        self.use_pert_idose = self.train_data.use_pert_idose
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size = self.batch_size, collate_fn = self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, collate_fn = self.collate_fn)

class DataReader(object):

    def __init__(self, drug_file, gene_file, data_file_train, data_file_dev, data_file_test,
                 data_filter, device, cell_ge_file_name):
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        self.gene = data_utils.read_gene(gene_file, self.device)
        feature_train, label_train, self.train_cell_type = data_utils.read_data(data_file_train, data_filter)
        feature_dev, label_dev, self.dev_cell_type = data_utils.read_data(data_file_dev, data_filter)
        feature_test, label_test, self.test_cell_type = data_utils.read_data(data_file_test, data_filter)
        self.train_feature, self.dev_feature, self.test_feature, self.train_label, \
        self.dev_label, self.test_label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev,
                                           feature_test, label_test, self.drug, self.device, cell_ge_file_name)

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
            cell_type = torch.Tensor(self.train_cell_type).to(self.device)
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
            cell_type = torch.Tensor(self.dev_cell_type).to(self.device)
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
            cell_type = torch.Tensor(self.test_cell_type).to(self.device)
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            index = index.numpy()
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = data_utils.convert_smile_to_feature(feature['drug'][excerpt], self.device)
            output['mask'] = data_utils.create_mask_feature(output['drug'], self.device)
            if self.use_pert_type:
                output['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output['pert_idose'] = feature['pert_idose'][excerpt]
            yield output, label[excerpt], cell_type[excerpt]

class AEDataReader(object):

    #### prepare L1000 gene expression profile as input
    #### prepare essential gene expression profile or L1000 gene expression profile as output
    def __init__(self, input_file_name, label_file_name, device):
        self.input_file_name = input_file_name
        self.label_file_name = label_file_name
        self.train_feature = torch.from_numpy(np.asarray(pd.read_csv(self.input_file_name + '_train.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.dev_feature = torch.from_numpy(np.asarray(pd.read_csv(self.input_file_name + '_dev.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.test_feature = torch.from_numpy(np.asarray(pd.read_csv(self.input_file_name + '_test.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.train_label = torch.from_numpy(np.asarray(pd.read_csv(self.label_file_name + '_train.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.dev_label = torch.from_numpy(np.asarray(pd.read_csv(self.label_file_name + '_dev.csv', index_col=0).values, dtype=np.float64)).to(device)
        self.test_label = torch.from_numpy(np.asarray(pd.read_csv(self.label_file_name + '_test.csv', index_col=0).values, dtype=np.float64)).to(device)

    #### get batch data: 
    #### input: dataset: indicate whether this is train, dev or test
    ####        batch_size: indicate batch size
    ####        shuffle: inidicate whether i need to shuffle the data
    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
        if shuffle:
            index = torch.randperm(len(feature)).long()
            index = index.numpy()
        for start_idx in range(0, len(feature), batch_size):
            if shuffle:
                excerpt = index[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield feature[excerpt], label[excerpt], torch.Tensor([*range(len(feature[excerpt]))]).long()

if __name__ == '__main__':
    data_filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
              "cell_id": ["A375", "HT29", "MCF7", "PC3", "HA1E", "YAPC", "HELA"],
              "pert_idose": ["0.04 um", "0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
