import torch
import torch.nn as nn
from neural_fingerprint import NeuralFingerprint
from drug_gene_attention import DrugGeneAttention
from ltr_loss import point_wise_mse, list_wise_listnet, list_wise_listmle, pair_wise_ranknet, list_wise_rankcosine, \
    list_wise_ndcg, combine_loss, mse_plus_homophily
import pdb
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GaussianNoise(nn.Module):
    '''Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    '''

    def __init__(self, sigma=0.1, is_relative_detach=True, device = 'cpu'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.std().detach() if self.is_relative_detach else self.sigma * x.std()
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

class LinearEncoder(nn.Module):

    '''
    the linear encoder for gene expression data
    :cell_id_input_dim: the dimension of gene expression profile
    '''
    def __init__(self, cell_id_input_dim):
        super(LinearEncoder, self).__init__()
        self.cell_id_1 = nn.Linear(cell_id_input_dim, 200)
        self.cell_id_2 = nn.Linear(200, 100)
        self.cell_id_3 = nn.Linear(100, 50)
        self.cell_id_embed_linear_only = nn.Sequential(self.cell_id_1, nn.ReLU(),
                                                    self.cell_id_2, nn.ReLU(),
                                                    self.cell_id_3, nn.ReLU())

    def forward(self, input_cell_gex, epoch = 0):
        '''
        :params: input_cell_gex: the gene expression profile
        : return: cell_hidden_: the hidden representation of each cell (batch * 50)
        '''
        if epoch % 100 == 80:
            print('------followings are multidcp before linear embed ----------')
            print(input_cell_gex)

        cell_hidden_ = self.cell_id_embed_linear_only(input_cell_gex)
        if epoch % 100 == 80:
            print('------followings are multidcp after linear embed ----------')
            print(cell_hidden_)
        # cell_hidden_ = [batch * cell_id_emb_dim(50)]
        return cell_hidden_

class TransformerEncoder(nn.Module):

    '''
    the transformer based encoder for gene expression data
    :cell_id_input_dim: the dimension of gene expression profile
    '''
    def __init__(self, cell_id_input_dim):
        super(TransformerEncoder, self).__init__()
        self.cell_id_embed = nn.Sequential(nn.Linear(cell_id_input_dim, 200), nn.Linear(200, 50))
        self.trans_cell_embed_dim = 32
        # self.cell_id_embed_1 = nn.Linear(1, self.trans_cell_embed_dim)
        self.cell_id_transformer = nn.Transformer(d_model = self.trans_cell_embed_dim, nhead = 4,
                                                num_encoder_layers = 1, num_decoder_layers = 1,
                                                dim_feedforward = self.trans_cell_embed_dim * 4)

        self.pos_encoder = PositionalEncoding(self.trans_cell_embed_dim)

    def forward(self, input_cell_gex, epoch = 0):
        '''
        :params: input_cell_gex: the gene expression profile
        : return: cell_hidden_: the hidden representation of each cell (batch * 50)
        '''
        cell_id_embed = self.cell_id_embed(input_cell_gex) # Transformer
        # cell_id_embed = [batch * 50]
        cell_id_embed = cell_id_embed.unsqueeze(-1)  # Transformer
        # cell_id_embed = [batch * 50 * 1]
        cell_id_embed = cell_id_embed.repeat(1,1,self.trans_cell_embed_dim)
        # cell_id_embed = self.cell_id_embed_1(cell_id_embed) # Transformer
        # cell_id_embed = [batch * 50 * 32(trans_cell_embed_dim)]
        if epoch % 100 == 80:
            print('------followings are multidcp before transformer ----------')
            print(cell_id_embed)
            torch.save(cell_id_embed, 'cell_id_embed_pre.pt')
        cell_id_embed = self.pos_encoder(cell_id_embed)
        cell_id_embed = self.cell_id_transformer(cell_id_embed, cell_id_embed) # Transformer
        # cell_id_embed = [batch * 50 * 32(trans_cell_embed_dim)]
        if epoch % 100 == 80:
            print('------followings are multidcp after transformer ----------')
            print(cell_id_embed)
            torch.save(cell_id_embed, 'cell_id_embed_post.pt')
        cell_hidden_, _ = torch.max(cell_id_embed, -1)
        # cell_hidden_ = [batch * 50]
        return cell_hidden_

class MultiDCP(nn.Module):

    '''
    The core module for multidcp.
    '''

    def __init__(self, device, model_param_registry):
        super(MultiDCP, self).__init__()
        self.set_attr_from_dict(model_param_registry)
        assert self.drug_emb_dim == self.gene_emb_dim, 'Embedding size mismatch'
        self.drug_fp = NeuralFingerprint(self.drug_input_dim['atom'], self.drug_input_dim['bond'], 
                                         self.conv_size, self.drug_emb_dim,
                                         self.degree, device)
        self.gene_embed = nn.Linear(self.gene_input_dim, self.gene_emb_dim)
        self.drug_gene_attn = DrugGeneAttention(self.gene_emb_dim, self.gene_emb_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=self.dropout, device=device)
        
        if self.linear_encoder_flag:
            self.encoder = LinearEncoder(self.cell_id_input_dim)
        else:
            self.encoder = TransformerEncoder(self.cell_id_input_dim)

        self.pert_idose_embed = nn.Linear(self.pert_idose_input_dim, self.pert_idose_emb_dim)
        self.linear_dim = self.drug_emb_dim + self.gene_emb_dim + self.cell_id_emb_dim + self.pert_idose_emb_dim

        self.linear_1 = nn.Linear(self.linear_dim, self.hid_dim)
        self.relu = nn.ReLU()

    def set_attr_from_dict(self, model_param_registry):

        for k, v in model_param_registry.items():
            self.__setattr__(k, v)

    def forward(self, input_drug, input_gene, mask, input_cell_gex, input_pert_idose, epoch = 0):
        # input_drug = {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}
        # gene_embed = [num_gene * gene_emb_dim]
        num_batch = input_drug['molecules'].batch_size

        ## drug embeding part
        drug_atom_embed = self.drug_fp(input_drug)
        # drug_atom_embed = [batch * num_node * drug_emb_dim]
        drug_embed = torch.sum(drug_atom_embed, dim=1)
        # drug_embed = [batch * drug_emb_dim]
        drug_embed = drug_embed.unsqueeze(1)
        # drug_embed = [batch * 1 *drug_emb_dim]
        drug_embed = drug_embed.repeat(1, self.num_gene, 1)
        # drug_embed = [batch * num_gene * drug_emb_dim]

        ## gene embedding part
        gene_embed = self.gene_embed(input_gene)
        # gene_embed = [num_gene * gene_emb_dim]
        gene_embed = gene_embed.unsqueeze(0)
        # gene_embed = [1 * num_gene * gene_emb_dim]
        gene_embed = gene_embed.repeat(num_batch, 1, 1)
        # gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed, _ = self.drug_gene_attn(gene_embed, drug_atom_embed, None, mask)
        # drug_gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed = torch.cat((drug_gene_embed, drug_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_emb_dim + gene_emb_dim)]
        
        ## cell line embedding part
        cell_hidden_ = self.encoder(input_cell_gex, epoch = epoch)
        # cell_hidden_ = [batch * 50]
        cell_id_embed = cell_hidden_.unsqueeze(1).repeat(1, self.num_gene, 1)
        # cell_id_embed = [batch * num_gene * 50]
        drug_gene_embed = torch.cat((drug_gene_embed, cell_id_embed), dim=2) 
        
        ## dosage embedding part
        pert_idose_embed = self.pert_idose_embed(input_pert_idose)
        # pert_idose_embed = [batch * pert_idose_emb_dim]
        pert_idose_embed = pert_idose_embed.unsqueeze(1)
        # pert_idose_embed = [batch * 1 * pert_idose_emb_dim]
        pert_idose_embed = pert_idose_embed.repeat(1, self.num_gene, 1)
        # pert_idose_embed = [batch * num_gene * pert_idose_emb_dim]

        drug_gene_embed = torch.cat((drug_gene_embed, pert_idose_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        drug_gene_embed = self.relu(drug_gene_embed)

        ## First layer of the mlp layers        
        out = self.linear_1(drug_gene_embed)
        # out = [batch * num_gene * hid_dim]
        return out, cell_hidden_

class MultiDCPBase(nn.Module):
    '''
    The bases module for multidcp. The following multiple variants for multidco are inherited from this module
    '''
    def __init__(self, device, model_param_registry):
        super(MultiDCPBase, self).__init__()
        self.multidcp = MultiDCP(device, model_param_registry)
        self.loss_type = model_param_registry['loss_type']
        self.initializer = model_param_registry['initializer']
        self.device = device

    def init_weights(self, pretrained = False):
        print('Initialized multidcp\'s weight............')
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 10**-7)
                else:
                    self.initializer(parameter)
        if pretrained:
            self.multidcp.load_state_dict(torch.load('best_model_ehill_storage_'))

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif self.loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif self.loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif self.loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        elif self.loss_type == 'combine':
            loss = combine_loss(label, predict, self.device)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss

class MultiDCPOriginal(MultiDCPBase):

    '''
    MultiDCP + relu + linear to predict the gene expression profile
    '''

    def __init__(self, device, model_param_registry):
        super(MultiDCPOriginal, self).__init__(device, model_param_registry)
        self.relu = nn.ReLU()
        self.linear_final = nn.Linear(model_param_registry['hid_dim'], 1)

    def forward(self, input_drug, input_gene, mask, input_cell_gex,
                input_pert_idose, epoch = 0):
        out = self.multidcp(input_drug, input_gene, mask, input_cell_gex,
                              input_pert_idose, epoch = epoch)[0]
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_final(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
        # out = [batch * num_gene]
        return out

class MultiDCPPretraining(MultiDCPBase):

    '''
    MultiDCP + two task specific linear to predict auc and ic50, which is used as a pretraining model
    '''

    def __init__(self, device, model_param_registry):
        super(MultiDCPPretraining, self).__init__(device, model_param_registry)
        self.relu = nn.ReLU()
        self.hid_dim = model_param_registry['hid_dim']
        self.task_1_linear = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim//2), 
                                            nn.ReLU(), 
                                            nn.Linear(self.hid_dim//2, self.hid_dim//2),
                                            nn.ReLU(),
                                            nn.Linear(self.hid_dim//2, 1)
                                            )
        self.task_2_linear = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim//2), 
                                            nn.ReLU(), 
                                            nn.Linear(self.hid_dim//2, self.hid_dim//2),
                                            nn.ReLU(),
                                            nn.Linear(self.hid_dim//2, 1)
                                            )

    def forward(self, input_drug, input_gene, mask, input_cell_gex, input_pert_idose):
        out = self.multidcp(input_drug, input_gene, mask, input_cell_gex, input_pert_idose)[0]
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = torch.sum(out, dim=1).squeeze(1)
        # out = [batch * 1 * hid_dim] => [batch * hid_dim] 
        out_1 = self.task_1_linear(out)
        # out_1 = [batch * 1] (pic50)
        out_2 = self.task_2_linear(out)
        # out_1 = [batch * 1] (auc)
        out = torch.cat((out_1, out_2), dim=1)
        # out = [batch * 2] (pic50 and auc)
        return out

class MultiDCPEhillPretraining(MultiDCPBase):

    '''
    MultiDCP + task specific linear to predict ehill (whether dosage specific?)
    '''

    def __init__(self, device, model_param_registry):
        super(MultiDCPEhillPretraining, self).__init__(device, model_param_registry)
        self.relu = nn.ReLU()
        self.hid_dim = model_param_registry['hid_dim']
        self.num_gene = model_param_registry['num_gene']
        self.task_linear = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim//2), 
                                            nn.ReLU(), 
                                            nn.Linear(self.hid_dim//2, self.hid_dim//2),
                                            nn.ReLU(),
                                            nn.Linear(self.hid_dim//2, 1))

        self.genes_linear = nn.Sequential(nn.Linear(self.num_gene, self.num_gene//2),
                                            nn.ReLU(),
                                            nn.Linear(self.num_gene//2, self.num_gene//2),
                                            nn.ReLU(),
                                            nn.Linear(self.num_gene//2, 1))
        self.linear_final = nn.Linear(self.hid_dim, 1)

    def forward(self, input_drug, input_gene, mask, input_cell_gex,
                input_pert_idose, job_id = 'perturbed', epoch = 0):

        out, cell_hidden_ = self.multidcp(input_drug, input_gene, mask, input_cell_gex,
                              input_pert_idose, epoch = epoch)

        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        if job_id == 'perturbed':
            out = self.linear_final(out)
            # out = [batch * num_gene * 1]
            out = out.squeeze(2)
            # out = [batch * num_gene]
        else:
            out = self.task_linear(out)
            # out = [batch * num_gene * 1]
            out = out.squeeze(2)
            # out = [batch * num_gene]
            out = self.genes_linear(out)
            # out = [batch*1] (ehill)

        return out, cell_hidden_

class MultiDCP_AE(MultiDCPBase):

    '''
    MultiDCP + encoder,decoder structure (if job id == 'perturbed", multidcp, else, encoder, decoder_only)
    '''
    def __init__(self, device, model_param_registry):
        super(MultiDCP_AE, self).__init__(device, model_param_registry)
        self.guassian_noise = GaussianNoise(device=device)
        self.relu = nn.ReLU()
        self.linear_final = nn.Linear(model_param_registry['hid_dim'], 1)
        self.decoder_linear = nn.Sequential(nn.Linear(50, 200), nn.Linear(200, model_param_registry['cell_decoder_dim']))

    def forward(self, input_cell_gex, input_drug = None, input_gene = None, mask = None, 
                input_pert_idose = None, job_id = 'perturbed', epoch = 0):
        if job_id == 'perturbed':
            return self.perturbed_trans(input_cell_gex, input_drug, input_gene, mask, 
                        input_pert_idose, epoch)
        else:
            return self.autoencoder(input_cell_gex, epoch)

    def perturbed_trans(self, input_cell_gex, input_drug, input_gene, mask, 
                        input_pert_idose, epoch):
        out, cell_hidden_ = self.multidcp(input_drug, input_gene, mask, input_cell_gex,
                                                input_pert_idose, epoch = epoch)
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_final(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
        # out = [batch * num_gene]
        return out, cell_hidden_

    def autoencoder(self, input_cell_gex, epoch = 0):
        ## autoencoder
        hidden = self.guassian_noise(input_cell_gex)
        # input_cell_gex = [batch * 978]
        cell_hidden_ = self.multidcp.encoder(hidden)
        # cell_hidden_ = [batch * cell_id_emb_dim(32)]
        out = self.decoder_linear(cell_hidden_)
                    
        if epoch % 100 == 80:
            print('---------------------followings are ae after transformer/linear embed in ae-------------------------')
            print(out)
        return out, cell_hidden_


