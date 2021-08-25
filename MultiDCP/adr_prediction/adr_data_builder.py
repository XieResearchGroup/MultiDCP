import pandas as pd
import random
from sklearn.model_selection import KFold, GroupKFold
import pdb

class ADRDataBuilder:

    def __init__(self, file_name):
        '''
        build the side effect dataframe from either FAERS_offsides_PTs.csv or *_PTs.csv
        :file_name: str
        '''
        self.side_effect_df = pd.read_csv(file_name, index_col = 0)
        # self.__remove_drugs_with_less_ADR()
        self.__remove_adrs_with_less_drugs()
        assert self.side_effect_df.shape[0] == len(set(self.side_effect_df.index)), "the pert id has duplications"
    
    def __remove_adrs_with_less_drugs(self):
        '''
        remove the adrs with very few drugs
        '''
        new_df = self.side_effect_df[self.get_side_effect_names()]
        sparse_filter = new_df.values.sum(axis = 0) > 10
        self.side_effect_df = self.side_effect_df.loc[: ,sparse_filter]

    def __remove_drugs_with_less_ADR(self):
        '''
        remove the drugs with very few ADRs
        '''
        new_df = self.side_effect_df[self.get_side_effect_names()]
        sparse_filter = new_df.values.sum(axis = 1) > 30
        self.side_effect_df = self.side_effect_df.loc[sparse_filter,:]

    def get_whole_df(self):
        return self.side_effect_df

    def get_side_effects_df_only(self):
        '''
        the return columns only have the differetn side effects
        '''
        return self.side_effect_df[self.get_side_effect_names()]

    def get_side_effect_names(self):
        cols = list(self.side_effect_df.columns)
        if 'pert_id' in cols:
            cols.remove('pert_id')
        return cols

    def get_drug_list(self):
        return list(self.side_effect_df.index)

    def prepare_adr_df_basedon_perts(self, pertid_list):
        '''
        prepare the side effect profile based on the pert id list
        :pertid_list: list
        :return: dataframe: the dataframe with index as pertid_list
        '''
        extra_pert = set(pertid_list) - set(self.get_drug_list())
        assert len(extra_pert) == 0, "there are pertid not found in the ADR file"
        return_cols = self.get_side_effect_names()
        return self.side_effect_df.loc[pertid_list, return_cols]

class PerturbedDGXDataBuilder:

    def __init__(self, gx_file_name, drug_cs_dir, pert_list, pred_flag=False, cs_part = True):
        '''
        build the pertubed gene expression dataframe from either FAERS_offsides_PTs_PredictionDGX.csv or *_PTs_PredictionDGX.csv
        :file_name: str
        :pred_flag: Boolean, whether the build dataframe is predicted DGX or groundtruth DGX
        '''
        self._pred_flag = pred_flag
        self.dgx_df = pd.read_csv(gx_file_name)
        self.cs_df = pd.read_csv(drug_cs_dir, index_col = 0)
        pertid_set = set(pert_list)
        self.x_ls = []
        self.dgx_df = self.dgx_df.loc[self.dgx_df['pert_id'].isin(pertid_set),:]
        self.x_ls.append(self.dgx_df)
        self.cs_df = self.cs_df.loc[self.dgx_df.pert_id, :]
        assert len(self.dgx_df) == sum(self.dgx_df.pert_id == self.cs_df.index), "dgx and cs df has different pert_ids "
        if cs_part:
            self.x_ls.append(self.cs_df.reset_index(drop = True))
        self.x_df = pd.concat(self.x_ls, axis = 1)
        # self.dgx_df = self.dgx_df.drop_duplicates('pert_id')

    def get_whole_df(self):
        return self.x_df

    def get_pred_flag(self):
        return self._pred_flag

    def get_filter_df(self, pertid_list):
        '''
        :pertid_list: list: a list of pert ids to filter the dataframe
        :return: dataframe: processed dataframe
        '''
        pertid_set = set(pertid_list)
        return self.x_df.loc[self.x_df['pert_id'].isin(pertid_set),:]
    
    def get_pert_id_list(self):
        return list(self.x_df.pert_id)

    def get_gx_only(self):
        '''
        the return colums only the gene expression features
        '''
        return self.x_df.iloc[:, 5:]

class XYPreparer:

    def __init__(self, X, Y, split_list, random_seed):

        self.X = X
        self.Y = Y
        self.split_list = split_list
        self.random_seed = random_seed
        
    def k_fold_split(self):
        kf = KFold(n_splits = 5, random_state = self.random_seed)
        for train_index, test_index in kf.split(self.X, self.Y):
            yield train_index, test_index

    def leave_new_drug_out_split(self):
        random.seed(self.random_seed)
        gkf = GroupKFold(n_splits = 5)
        for train_index, test_index in gkf.split(self.X, None, self.split_list):
            yield train_index, test_index
