
import numpy as np
import random
import torch
from molecules import Molecules
import pdb
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
warnings.filterwarnings("ignore")


def read_drug_number(input_file, num_feature):
    drug = []
    drug_vec = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            assert len(line) == num_feature + 1, "Wrong format"
            bin_vec = [float(i) for i in line[1:]]
            drug.append(line[0])
            drug_vec.append(bin_vec)
    drug_vec = np.asarray(drug_vec, dtype=np.float64)
    index = []
    for i in range(np.shape(drug_vec)[1]):
        if len(set(drug_vec[:, i])) > 1:
            index.append(i)
    drug_vec = drug_vec[:, index]
    drug = dict(zip(drug, drug_vec))
    return drug, len(index)

def read_drug_string(input_file):
    with open(input_file, 'r') as f:
        drug = dict()
        for line in f:
            line = line.strip().split(',')
            assert len(line) == 2, "Wrong format"
            drug[line[0]] = line[1]
    return drug, None


def read_gene(input_file, device):
    with open(input_file, 'r') as f:
        gene = []
        for line in f:
            line = line.strip().split(',')
            assert len(line) == 129, "Wrong format"
            gene.append([float(i) for i in line[1:]])
    return torch.from_numpy(np.asarray(gene, dtype=np.float64)).to(device)


def convert_smile_to_feature(smiles, device):
    molecules = Molecules(smiles)
    node_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('atom')]).to(device).double()
    edge_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('bond')]).to(device).double()
    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}


def create_mask_feature(data, device):
    batch_idx = data['molecules'].get_neighbor_idx_by_batch('atom')
    molecule_length = [len(idx) for idx in batch_idx]
    mask = torch.zeros(len(batch_idx), max(molecule_length)).to(device).double()
    for idx, length in enumerate(molecule_length):
        mask[idx][:length] = 1
    return mask


def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]


def split_data_by_pert_id(pert_id):
    random.shuffle(pert_id)
    num_pert_id = len(pert_id)
    fold_size = int(num_pert_id/10)
    train_pert_id = pert_id[:fold_size*6]
    dev_pert_id = pert_id[fold_size*6: fold_size*8]
    test_pert_id = pert_id[fold_size*8:]
    return train_pert_id, dev_pert_id, test_pert_id

def get_class_vote(pert_list, bottom_threshold, top_threshold):
    votes = [0, 0, 0, 0]
    # list of perts
    for pert in pert_list:
        if pert > top_threshold:
            votes[3] += 1  # upregulation
        elif pert < bottom_threshold:
            votes[1] += 1  # downregulation
        else:
            votes[2] += 1  # not regulated
    highest_vote_class = np.argmax(votes)
    is_tie = False  # check if there's a tie (another class with the same number of votes)
    for i in range(0, len(votes)):
        if i == highest_vote_class:
            continue
        if votes[i] == votes[highest_vote_class]:
            is_tie = True
            break
    if is_tie:
        return 0
    else:
        return highest_vote_class



def read_data(input_file, filter):
    """
    :param input_file: including the time, pertid, perttype, cellid, dosage and the perturbed gene expression file (label)
    :param filter: help to check whether the pertid is in the research scope, cells in the research scope ...
    :return: the features, labels and cell type
    """
    feature = []
    labels = []
    
    data = dict()
    pert_id = []
    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split(',')
           
            ft = ','.join(line[0:4])
            lb = [i for i in line[4:]]
            if ft in data.keys():
                data[ft].append(lb)
            else:
                data[ft] = [lb]
                    
    for ft, lb in sorted(data.items()):
    
        ft = ft.split(',')
        feature.append(ft)
        labels.append(lb[0])
    
    return np.asarray(feature), np.asarray(labels,dtype=np.float64)

def transform_to_tensor_per_dataset(feature, label, drug,device, basal_expression_file):

    """
    :param feature: features like pertid, dosage, cell id, etc. will be used to transfer to tensor over here
    :param label:
    :param drug: ??? a drug dictionary mapping drug name into smile strings
    :param device: save on gpu device if necessary
    :return:
    """

    if not basal_expression_file.endswith('csv'):
        basal_expression_file += '.csv'
    basal_cell_line_expression_feature_csv = pd.read_csv(basal_expression_file, index_col = 0)
    drug_feature = []
    drug_target_feature = []
    pert_type_set = sorted(list(set(feature[:, 1])))
    cell_id_set = sorted(list(set(feature[:, 2])))
    pert_idose_set = sorted(list(set(feature[:, 3])))
    # pert_type_set = ['trt_cp']
    # cell_id_set = ['HA1E', 'HT29', 'MCF7', 'YAPC', 'HELA', 'PC3', 'A375']
    # pert_idose_set = ['1.11 um', '0.37 um', '10.0 um', '0.04 um', '3.33 um', '0.12 um']
    use_pert_type = False
    use_cell_id = True ## cell feature will always used
    use_pert_idose = False
    if len(pert_type_set) > 1:
        pert_type_dict = dict(zip(pert_type_set, list(range(len(pert_type_set)))))
        final_pert_type_feature = []
        use_pert_type = True
    if len(cell_id_set) > 1:
        cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
        final_cell_id_feature = []
        use_cell_id = True
    if len(pert_idose_set) > 1:
        pert_idose_dict = dict(zip(pert_idose_set, list(range(len(pert_idose_set)))))
        final_pert_idose_feature = []
        use_pert_idose = True
    print('Feature Summary (printing from data_utils):')
    print(pert_type_set)
    print(cell_id_set)
    print(pert_idose_set)

    for i, ft in enumerate(feature):
        drug_fp = drug[ft[0]]
        drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[1]]] = 1
            final_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float64))
        if use_cell_id:
            # cell_id_feature = np.zeros(len(cell_id_set))
            # cell_id_feature[cell_id_dict[ft[2]]] = 1
            cell_id_feature = basal_cell_line_expression_feature_csv.loc[ft[2],:] ## new_code
            final_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float64))
      
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            final_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float64))
      

    feature_dict = dict()
    feature_dict['drug'] = np.asarray(drug_feature)
    if use_pert_type:
        feature_dict['pert_type'] = torch.from_numpy(np.asarray(final_pert_type_feature, dtype=np.float64)).to(device)
    if use_cell_id:
        feature_dict['cell_id'] = torch.from_numpy(np.asarray(final_cell_id_feature, dtype=np.float64)).to(device)
    if use_pert_idose:
        feature_dict['pert_idose'] = torch.from_numpy(np.asarray(final_pert_idose_feature, dtype=np.float64)).to(device)
    label_binary = torch.from_numpy(label).to(device)
    return feature_dict, label_binary, use_pert_type, use_cell_id, use_pert_idose


def transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev, feature_test, label_test, drug,
                        device, basal_expression_file_name):

    """
    :param feature_train: features like pertid, dosage, cell id, etc. will be used to transfer to tensor over here
    :param label_train:
    :param feature_dev:
    :param label_dev:
    :param feature_test:
    :param label_test:
    :param drug: ??? a drug dictionary mapping drug name into smile strings
    :param device: save on gpu device if necessary
    :return:
    """
    # basal_expression_file_name = 'ccle_gene_expression_file.csv'
    # basal_expression_file_name = 'ccle_gene_expression_2176.csv'
    train_feature, train_label_regression, use_pert_type_train, use_cell_id_train, use_pert_idose_train = \
        transform_to_tensor_per_dataset(feature_train, label_train, drug, device, basal_expression_file_name)
    dev_feature, dev_label_regression, use_pert_type_dev, use_cell_id_dev, use_pert_idose_dev = \
        transform_to_tensor_per_dataset(feature_dev, label_dev, drug, device, basal_expression_file_name)
    test_feature, test_label_regression, use_pert_type_test, use_cell_id_test, use_pert_idose_test = \
        transform_to_tensor_per_dataset(feature_test, label_test, drug, device, basal_expression_file_name)
    assert use_pert_type_train == use_pert_type_dev and use_pert_type_train == use_pert_type_test, \
            'use pert type is not consistent'
    assert use_cell_id_train == use_cell_id_dev and use_cell_id_train == use_cell_id_test, \
            'use cell id is not consistent'
    assert use_pert_idose_train == use_pert_idose_dev and use_pert_idose_train == use_pert_idose_test, \
            'use pert idose is not consistent'
    return train_feature, dev_feature, test_feature, train_label_regression, dev_label_regression, \
           test_label_regression, use_pert_type_train, use_cell_id_train, use_pert_idose_train

def binary_transfer(input_file,save_path_up,save_path_down):
    '''
    use this func for binary setting data process only 
    
    '''

    
    filter = {"time": "24H",
            "pert_type": ["trt_cp"],
            "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
    feature = []
    label = []
    data = dict()
    pert_id = []
    
    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split(',')
            # assert len(line) == 983 or len(line) == 7 or len(line) == 6, "Wrong format"
            if filter["time"] in line[0]  and line[2] in filter["pert_type"] \
                    and line[4] in filter["pert_idose"]: # filter["time"] in line[0] and 
                ft = ','.join(line[1:5])
                lb = [float(i) for i in line[5:]]
                if ft in data.keys():
                    data[ft].append(lb)
                else:
                    data[ft] = [lb]

    for ft, lb in sorted(data.items()):
        ft = ft.split(',')
        feature.append(ft)
        
        pert_id.append(ft[0])
        if len(lb) == 1:
            label.append(lb[0])
        else:
            lb = choose_mean_example(lb)
            label.append(lb)
   
    label = np.asarray(label, dtype=np.float64)
    feature= np.asarray(feature)
    label_sort_set = np.argsort(label,axis=1)
    neg_set = label_sort_set[:,:50]   # use 5% ranking as neg/pos 
    pos_set = label_sort_set[:, -50:]
    down_label = np.zeros([len(label),978])
    up_label = np.zeros([len(label),978])
    down_label[np.arange(down_label.shape[0])[:,None],neg_set] =1
    up_label[np.arange(up_label.shape[0])[:,None],pos_set] =1

    sorted_test_input = pd.read_csv(input_file).sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])

    genes_cols = sorted_test_input.columns[5:]
    #genes_cols = all_genes

    drug_id = feature[:,0]
    UP_ground_truth_df = pd.DataFrame(up_label, index =drug_id, columns = genes_cols)
    #up_pos = np.count_nonzero(UP_ground_truth_df)
    
    UP_ground_truth_df.insert(0, 'pert_type', feature[:,1])
    UP_ground_truth_df.insert(1, 'cell_id', feature[:,2])
    UP_ground_truth_df.insert(2, 'pert_idose', feature[:,3])
    DOWN_ground_truth_df = pd.DataFrame(down_label, index =drug_id, columns = genes_cols)
    #down_pos = np.count_nonzero(DOWN_ground_truth_df)
    DOWN_ground_truth_df.insert(0, 'pert_type', feature[:,1])
    DOWN_ground_truth_df.insert(1, 'cell_id', feature[:,2])
    DOWN_ground_truth_df.insert(2, 'pert_idose', feature[:,3])

    UP_ground_truth_df.to_csv(save_path_up)
    DOWN_ground_truth_df.to_csv(save_path_down)
    #print(f'up pos and down pos is {up_pos},{down_pos}')
    print(f'finish processing')

if __name__ == '__main__':
    binary_transfer(input_file='/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/pert_transcriptom/signature_dev_cell_3.csv',
                    save_path_down='/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_dev_3.csv',
                    save_path_up='/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/up_signature_dev_3.csv')