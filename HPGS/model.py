import math

import numpy as np
import torch

from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops,softmax

class GCNConv_SH(MessagePassing):
    def __init__(self, in_channels, out_channels,alpha,drop,node_dim = 0):
        super(GCNConv_SH, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()
        self.node_dim = node_dim
        self.alpha =alpha
        self.drop = drop
        self.leakralu = nn.LeakyReLU(alpha)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_channels, 1)))
        nn.init.xavier_uniform_(self.a)
    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        h = self.lin(x)
        h_prime = self.propagate(edge_index, x=h)
        return h_prime

    def message(self, x_i, x_j, edge_index_i):
        e = torch.matmul((torch.cat([x_i, x_j], dim=-1)), self.a)
        e = self.leakralu(e)
        alpha = softmax(e, edge_index_i)
        alpha = F.dropout(alpha, self.drop,self.training)
        return x_j * alpha
class GCNConv_SS_HH(MessagePassing):
    def __init__(self, in_channels, out_channels,alpha,dorp,node_dim = 0):
        super(GCNConv_SS_HH, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()
        self.a = nn.Parameter(torch.zeros(size=(2*out_channels,1)))
        self.leakralu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.a)
        self.node_dim = node_dim
        self.alpha = alpha
        self.drop = dorp
    def forward(self, x, edge_index):
        edge_index,_ = add_remaining_self_loops(edge_index)
        h = self.lin(x)
        h_prime = self.propagate(edge_index,x = h)
        return h_prime
    def message(self,x_i,x_j,edge_index_i):
        e = torch.matmul((torch.cat([x_i,x_j],dim=-1)),self.a)
        e = self.leakralu(e)
        alpha = softmax(e,edge_index_i)
        alpha = F.dropout(alpha,self.drop,self.training)
        return x_j*alpha
class RETAIN(nn.Module):
    def __init__(self, emb_dim):
        super(RETAIN, self).__init__()
        self.emb_dim = emb_dim
        self.encoder = nn.GRU(emb_dim, emb_dim * 2, batch_first=True)
        self.alpha_net = nn.Linear(emb_dim * 2, 1)
        self.beta_net = nn.Linear(emb_dim * 2, emb_dim)
    def forward(self, i1_seq):
        o1, h1 = self.encoder(i1_seq)
        ej1 = self.alpha_net(o1)
        bj1 = self.beta_net(o1)
        att_ej1 = torch.softmax(ej1, dim=1)
        o1 = att_ej1 * torch.tanh(bj1) * i1_seq
        return o1,h1
class HM_PGS(nn.Module):
    def __init__(self,sh_num,voc_size,emb_dim = 64,device = torch.device('cuda'),batchsize=1,drop=0.5):

        super(HM_PGS,self).__init__()
        self.vocab_size =voc_size
        self.device = device
        self.batchsize = batchsize
        self.drop = drop
        self.SH_embedding = torch.nn.Embedding(sh_num,emb_dim)
        self.convSH_Tostudys1 = GCNConv_SH(emb_dim,emb_dim,drop=drop,alpha=0.5)
        self.convSH_Tostudys2 = GCNConv_SH(emb_dim, emb_dim,drop=drop,alpha=0.5)
        self.SHmlp=torch.nn.Linear(emb_dim,256)
        self.shbn_1 = torch.nn.BatchNorm1d(256)
        self.SH_TAN = torch.nn.Tanh()
        self.convSH_TostudyS_1_h = GCNConv_SH(emb_dim, emb_dim,alpha=0.5,drop=drop)
        self.convSH_TostudyS_2_h = GCNConv_SH(emb_dim,emb_dim,alpha=0.5,drop=drop)
        self.SH_mlp_1_h = torch.nn.Linear(emb_dim, 256)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1_h = torch.nn.Tanh()
        self.convSS = GCNConv_SS_HH(emb_dim, 128,dorp=0.2,alpha=0.5)
        self.convHH = GCNConv_SS_HH(emb_dim, 256,alpha=0.5,dorp=drop)
        self.conv_ddi = GCNConv_SS_HH(emb_dim,256,alpha=0.5,dorp=drop)
        self.S_256 = torch.nn.Linear(128, 256)
        self.mlp = torch.nn.Linear(256, 256)
        self.SI_bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.retain = nn.ModuleList([RETAIN(256) for _ in range(2)])
        self.getmedecine=nn.Linear(64,956)
        self.herbs = nn.Linear(256,956)
        self.L = nn.Linear(576,956)
        self.K = nn.Linear(956,256)
        self.P =nn.Linear(512,256)
        self.med_embedding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,emb_dim),
            nn.Dropout(0.5)
        )
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.nhead = 2
        self.medication_encoder = nn.TransformerEncoderLayer(64, self.nhead, dropout=0.2)
        self.MED_PAD_TOKEN = voc_size[1] + 2
        self.med_embedding = nn.Linear(512,emb_dim)
        self.tran = nn.Linear(4096, 956)
        self.abc = nn.Linear(256*2,256)

        self.sample = nn.Linear(256, 1)
        self.sample2 = nn.Linear(956, 1)
        self.sample3 = nn.Linear(112, 1)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256 * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, voc_size[1])
        )

        self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], 256) for i in range(2)])

    def calc_cross_visit_scores(self, visit_diag_embedding):
        max_visit_num = visit_diag_embedding.size(1)
        batch_size = visit_diag_embedding.size(0)
        new = visit_diag_embedding.size(2)
        mask = (torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        padding = torch.zeros((batch_size, 1, new), device=self.device).float()
        diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)
        diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) \
                      / math.sqrt(visit_diag_embedding.size(-1))
        scores = F.softmax(diag_scores + mask, dim=-1)
        return scores

    def forward(self, x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH,hd_x,hd_edge, input):
        def mean_embedding(embedding):
            return  embedding.mean(dim=1).unsqueeze(dim=0)
        if len(input)==0:
            raise Exception("Input error")
        def getindex(list,nums):
            x = [0]*nums
            for d in list:
                x[d]=1
            x = torch.LongTensor(x)
            return x
        x_SH1 = self.SH_embedding(x_SH.long())
        x_SH2 = self.convSH_Tostudys1(x_SH1.float(),edge_index_SH)
        x_SH3 =self.convSH_Tostudys2(x_SH2,edge_index_SH)
        x_sh4 = (x_SH1+x_SH2+x_SH3)/3
        x_SH5 = self.SHmlp(x_sh4)
        x_SH5=x_SH5.view(2322,-1)
        x_SH5 = self.shbn_1(x_SH5)
        x_SH5 = self.SH_TAN(x_SH5)
        x_SH6  = self.SH_embedding(x_SH.long())
        x_SH7 = self.convSH_TostudyS_1_h(x_SH6.float(),edge_index_SH)
        x_SH8 = self.convSH_TostudyS_2_h(x_SH7,edge_index_SH)
        x_SH9 = (x_SH6+x_SH7+x_SH8)/3.0
        x_SH9=self.SH_mlp_1_h(x_SH9)
        x_SH9=x_SH9.view(2322,-1)
        x_SH9 = self.SH_bn_1_h(x_SH9)
        x_SH9 = self.SH_tanh_1_h(x_SH9)
        x_ss0 = self.SH_embedding(x_SS.long())
        x_ss1 = self.convSS(x_ss0.float(),edge_index_SS)
        x_ss1 =x_ss1.view(1366,-1)
        x_hh0 = self.SH_embedding(x_HH.long())
        x_hh0 = x_hh0.view(-1,64)
        x_hh1 = self.convHH(x_hh0.float(),edge_index_HH)
        x_hh1 = x_hh1.view(956,-1)
        x_ss1 = self.S_256(x_ss1)
        x_hddi = self.conv_ddi(x_hh0,hd_edge)
        es = (x_SH5[:1366]+x_ss1)/2.0
        eh = (x_SH9[1366:]+x_hh1)/2.0-0.1*x_hddi
        i1_seq =[]
        i2_seq = []
        for adm in input:
            i1 = self.dropout(torch.mm(getindex(adm[0],1366).unsqueeze(dim=0).float().to(self.device),es).to(self.device))
            i2 = torch.mm(getindex(adm[1],956).unsqueeze(dim=0).float().to(self.device),eh)
            i1_seq.append(mean_embedding(i1.unsqueeze(dim=0)))
            i2_seq.append(mean_embedding(i2.unsqueeze(dim = 0)))
        i1_seq = torch.cat(list(reversed(i1_seq)), dim=1)
        o1,h1 = self.retain[0](i1_seq)
        visit_diag_embedding = o1.view(64, 1, -1)
        cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding)
        prob_g = F.softmax(cross_visit_scores, dim=-1)
        patient_representations = torch.cat([o1,o1],dim=-1).squeeze(dim=0)
        queries = self.query(patient_representations)
        query = queries[-1:]
        history_values = np.zeros((len(input) - 1, 956))
        b= torch.zeros(1,956).to(self.device)
        if len(input) > 1:
            i2_seq = torch.cat(list(reversed(i2_seq[:-1])),dim=1)
            o2,h2 = self.retain[1](i2_seq)
            last_seq_medication = h2
            last_seq_medication_emb = self.med_embedding(last_seq_medication)
            encoded_medication = self.medication_encoder(last_seq_medication_emb, src_mask=None)
            prob_g = prob_g.squeeze(dim=1)
            history_keys = query
            b = prob_g * encoded_medication.squeeze(dim=1)
            b = b.view(1, -1)
            b = self.tran(b)
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[1]] = 1
                history_items = torch.mm(torch.Tensor(history_values).to(self.device),eh)
            history_values = torch.FloatTensor(history_values).to(self.device)
        fun2 = torch.mm(query, eh.t())
        key_weights1 = F.softmax(torch.mm(query, eh.t()), dim=-1)
        fact1 = torch.mm(key_weights1, eh)
        w1 = torch.sigmoid(fun2)
        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)]
            visit_weight = F.softmax(torch.mm(query, history_keys.t()),dim=-1)
            weighted_values = visit_weight.mm(history_values)
            fact2 = torch.mm(weighted_values, eh)
        else:
            fact2 = fact1
        output = self.output(torch.cat([query, fact1, fact2], dim=-1))
        a = w1 * (0.9 * output) + 1.85 * b
        ehr_graph = self.sample(eh)
        ddi_graph = self.sample(x_hddi)
        fin_e = ehr_graph.t()
        fin_d = ddi_graph.t()
        return a,fin_e,fin_d



