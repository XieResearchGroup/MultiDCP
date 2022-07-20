import numpy as np
import random
import torch
from molecules import Molecules
import pdb
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
warnings.filterwarnings("ignore")
filter = {"time": "24H", 
            "pert_type": ["trt_cp"],
            #  "pert_idose":    ["10.0 um"]} 
            "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
                           

def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]
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

feature = []
label = []
UP_label = []
DOWN_label =[] 
data = dict()
pert_id = []
input_file = "MultiDCP_data/data/pert_transcriptom/signature_test_cell_1.csv"
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

##----------------------- for ranking test ----------------------
for ft, lb in sorted(data.items()):
    ft = ft.split(',')
    feature.append(ft)
    
    pert_id.append(ft[0])
    if len(lb) == 1:
        label.append(lb[0])
    else:
        lb = choose_mean_example(lb)
        label.append(lb)
_, cell_type = np.unique(np.asarray([x[2] for x in feature]), return_inverse=True)
label = np.asarray(label, dtype=np.float64)
feature= np.asarray(feature)
label_sort_set = np.argsort(label,axis=1)
neg_set = label_sort_set[:,:50]
pos_set = label_sort_set[:, -50:]
down_label = np.zeros([len(label),978])
up_label = np.zeros([len(label),978])
down_label[np.arange(down_label.shape[0])[:,None],neg_set] =1
up_label[np.arange(up_label.shape[0])[:,None],pos_set] =1

sorted_test_input = pd.read_csv("MultiDCP_data/data/pert_transcriptom/signature_test_cell_1.csv").sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])

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

UP_ground_truth_df.to_csv('/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/up_signature_test_1.csv')
DOWN_ground_truth_df.to_csv('/raid/home/yoyowu/MultiDCP/MultiDCP_data/ranking_binary/down_signature_test_1.csv')
#print(f'up pos and down pos is {up_pos},{down_pos}')
print(f'finish processing')
## in  data_utils 
    # label_sort_set = np.argsort(label,axis=1)
    # neg_set = label_sort_set[:,: 50]
    # pos_set = label_sort_set[:, -50:]
    # down_label = np.zeros([len(label),978])
    # up_label = np.zeros([len(label),978])
    # down_label[np.arange(down_label.shape[0])[:,None],neg_set] =1
    # up_label[np.arange(up_label.shape[0])[:,None],pos_set] =1
    # return np.asarray(feature),down_label, cell_type
###----------------------- end  ranking test ----------------------


##------------------ process test data for deepCOP
## if use our total data 
# totaldata=pd.read_csv('/raid/home/yoyowu/MultiDCP/MultiDCP_data/data/pert_transcriptom/signature_total.csv',index_col=0)
# print(f'the size of MultiDCP total signiture data is {len(totaldata)}, expected to be same as phase2 data used by deepCOP')
# totaldata=totaldata.drop(['cell_id','pert_idose','pert_type'],axis=1)
# totaldata=totaldata.set_index(['pert_id'])

# gene_cutoffs_down={}
# gene_cutoffs_up={}
# percentile_down = 5
# percentile_up = 100-percentile_down

# for gene in totaldata.columns:
#     row = totaldata[str(gene)]
#     gene_cutoffs_down[gene] = np.percentile(row, percentile_down)
#     gene_cutoffs_up[gene] = np.percentile(row, percentile_up)

# assert(len(gene_cutoffs_down)==978)
# all_genes = list(totaldata.columns)



## ------ prob no longer use 
# new_test_down = pd.DataFrame()
# new_test_up = pd.DataFrame()
# for gene in all_genes:    
#     new_test_down[gene] = testdata[gene]<gene_cutoffs_down[gene]
#     new_test_up[gene] = testdata[gene]>gene_cutoffs_up[gene]  

# -------  prob no longer use end --------
##-------use deepcop cutoffs ------------
# gene_cutoffs_down = np.load('/raid/home/yoyowu/MultiDCP/MultiDCP/utils/deepCOP_gene_cutoffs_down_ourgenes.npy',allow_pickle=True).item()
# gene_cutoffs_up = np.load('/raid/home/yoyowu/MultiDCP/MultiDCP/utils/deepCOP_gene_cutoffs_up_ourgenes.npy',allow_pickle=True).item()
# all_genes = list(gene_cutoffs_down.keys())
# #assert('WASHC4' in all_genes)
# for ft, lb in sorted(data.items()):
#     lb = np.array(lb)
#     ft = ft.split(',')
#     feature.append(ft)
#     #pert_id.append(ft[0])
#     lb_voted = np.zeros(978,dtype=int)
#     down_label = np.zeros(978,dtype=int)
#     up_label =  np.zeros(978,dtype=int)
#     for gene in all_genes:
#         lb_voted[all_genes.index(gene)] = get_class_vote(lb[:,all_genes.index(gene)], gene_cutoffs_down[gene], gene_cutoffs_up[gene])
 
#     down_locations = np.where(lb_voted == 1)
#     mid_locations = np.where(lb_voted == 2)
#     up_locations = np.where(lb_voted== 3)
#     down_label[down_locations] =1 
#     up_label[up_locations]=1
#     UP_label.append(up_label)
#     DOWN_label.append(down_label)


# feature= np.asarray(feature)
# UP_label = np.asarray(UP_label, dtype=np.float64)
# DOWN_label = np.asarray(DOWN_label, dtype=np.float64)
# print(f'the test data size:{len(UP_label)}')


# sorted_test_input = pd.read_csv("MultiDCP_data/data/pert_transcriptom/signature_test_cell_1.csv").sort_values(['pert_id', 'pert_type', 'cell_id', 'pert_idose'])

# #genes_cols = sorted_test_input.columns[5:]
# genes_cols = all_genes

# drug_id = feature[:,0]
# UP_ground_truth_df = pd.DataFrame(UP_label, index =drug_id, columns = genes_cols)
# up_pos = np.count_nonzero(UP_ground_truth_df)

# UP_ground_truth_df.insert(0, 'pert_type', feature[:,1])
# UP_ground_truth_df.insert(1, 'cell_id', feature[:,2])
# UP_ground_truth_df.insert(2, 'pert_idose', feature[:,3])
# DOWN_ground_truth_df = pd.DataFrame(DOWN_label, index =drug_id, columns = genes_cols)
# down_pos = np.count_nonzero(DOWN_ground_truth_df)
# DOWN_ground_truth_df.insert(0, 'pert_type', feature[:,1])
# DOWN_ground_truth_df.insert(1, 'cell_id', feature[:,2])
# DOWN_ground_truth_df.insert(2, 'pert_idose', feature[:,3])

# UP_ground_truth_df.to_csv('/raid/home/yoyowu/MultiDCP/DeepCOP_data/good_genes_up_multiDCP_testdata_w_deepcopCut.csv')
# DOWN_ground_truth_df.to_csv('/raid/home/yoyowu/MultiDCP/DeepCOP_data/good_genes_down_multiDCP_testdata_w_deepcopCut.csv')
# print(f'up pos and down pos is {up_pos},{down_pos}')