import math

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder

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
class NCModel(nn.Module):
    def __init__(self, n_nodes,output_dim):
        super(NCModel, self).__init__()
        self.c = torch.tensor([2.])
        self.manifold= 'PoincareBall'
        self.encoder = getattr(encoders, 'HGN_PGS')(self.c)
        self.decoder = model2decoder['HGN_PGS'](self.c)

    def encode(self, x, adj):
        if self.manifold == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output
    def compute_metrics(self, embeddings, data):
        output = self.decode(embeddings,data)
        return output
class HM_PGS(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, batchSize):
        super(HM_PGS, self).__init__()
        self.batchSize = batchSize
        self.voc_size=(ss_num,hh_num)
        self.SH_embedding = torch.nn.Embedding(sh_num, embedding_dim)
        self.SS_embedding = torch.nn.Embedding(ss_num,embedding_dim)
        self.HH_embedding = torch.nn.Embedding(sh_num,embedding_dim)
        self.HGN_PGSSS_Embedding=NCModel(n_nodes=ss_num,output_dim=embedding_dim)
        self.HGN_PGSSH_Embedding = NCModel(n_nodes=sh_num,output_dim=embedding_dim)
        self.HGN_PGSSH_Embedding2 = NCModel(n_nodes=sh_num, output_dim=embedding_dim)
        self.HGN_PGSHH_Embedding=NCModel(n_nodes=hh_num,output_dim=embedding_dim)
        self.sh_ems=torch.nn.Linear(16,16)
        self.SH_mlp_1 = torch.nn.Linear(16, 16)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(16)
        self.SH_tanh_1_h = torch.nn.Tanh()
        self.concatsLinear = torch.nn.Linear(16*2,16)
        self.concathLinear = torch.nn.Linear(16*2,16)
        self.Ls = torch.nn.Linear(16,16)
        self.mlp = torch.nn.Linear(16, 16)
        self.SI_bn = torch.nn.BatchNorm1d(16)
        self.relu = torch.nn.ReLU()
        self.conv_ddi=NCModel(n_nodes=hh_num,output_dim=embedding_dim)
        self.retain = nn.ModuleList([RETAIN(256) for _ in range(2)])
        self.getmedecine = nn.Linear(64, 379)
        self.herbs = nn.Linear(256, 379)
        self.L = nn.Linear(576, 379)
        self.K = nn.Linear(379, 256)
        self.P = nn.Linear(512, 256)
        self.med_embedding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.Dropout(0.5)
        )
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.nhead = 2
        self.medication_encoder = nn.TransformerEncoderLayer(64, self.nhead, dropout=0.2)
        self.MED_PAD_TOKEN = hh_num + 2
        self.med_embedding = nn.Linear(512, 64)
        self.tran = nn.Linear(4096, 379)
        self.abc = nn.Linear(256 * 2, 256)
        self.sample = nn.Linear(256, 1)
        self.sample2 = nn.Linear(16, 1)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256*2 , embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, hh_num)
        )
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.voc_size[i], 256) for i in range(2)])
        self.xh_encode=nn.Linear(16,256)
        self.xs_encode = nn.Linear(16,256)
        self.dropout = nn.Dropout(0.7)
        self.device = 'cuda'
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
    def forward(self, x_SH, Spare_SH,x_SS,Spare_SS,x_HH,Spare_HH,Spare_ddi, input):
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
        x_SH2 = self.HGN_PGSSH_Embedding.encode(x_SH1.squeeze(), Spare_SH)
        x_SH2=self.HGN_PGSSH_Embedding.compute_metrics(x_SH2,Spare_SH)
        x_SH3 = self.sh_ems(x_SH1.squeeze())
        x_SH2 = (x_SH2+x_SH3)/2.0
        x_SH2 = self.SH_mlp_1(x_SH2)
        x_SH2=self.SH_bn_1_h(x_SH2)
        x_SH2 = self.SH_tanh_1_h(x_SH2)
        x_ss1 = self.SS_embedding(x_SS.long())
        x_ss2 = self.HGN_PGSSS_Embedding.encode(x_ss1.squeeze(), Spare_SS)
        x_ss2 = self.HGN_PGSSS_Embedding.compute_metrics(x_ss2, Spare_SS)
        x_hh1 = self.HH_embedding(x_HH.long())
        x_hh2 = self.HGN_PGSHH_Embedding.encode(x_hh1.squeeze(),Spare_HH)
        x_hh2 = self.HGN_PGSHH_Embedding.compute_metrics(x_hh2,Spare_HH)
        x_hh2 = self.Ls(x_hh2)
        x_hddi = self.conv_ddi.encode(x_hh1.squeeze(), Spare_ddi)
        x_ddi = self.conv_ddi.compute_metrics(x_hddi,Spare_ddi)
        es = (x_SH2[:974] + x_ss2) / 2.0
        eh = (x_SH2[974:] + x_hh2) / 2.0-0.1*x_ddi
        es = self.xs_encode(es)
        eh = self.xh_encode(eh)
        i1_seq = []
        i2_seq = []
        for adm in input:
            i1 = self.dropout(
                torch.mm(getindex(adm[0], 974).unsqueeze(dim=0).float().to(self.device), es).to(self.device))
            i2 = torch.mm(getindex(adm[1], 379).unsqueeze(dim=0).float().to(self.device), eh)
            i1_seq.append(mean_embedding(i1.unsqueeze(dim=0)))
            i2_seq.append(mean_embedding(i2.unsqueeze(dim=0)))
        i1_seq = torch.cat(list(reversed(i1_seq)), dim=1)
        o1, h1 = self.retain[0](i1_seq)
        visit_diag_embedding = o1.view(64, 1, -1)
        cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding)
        prob_g = F.softmax(cross_visit_scores, dim=-1)
        patient_representations = torch.cat([o1, o1], dim=-1).squeeze(dim=0)
        queries = self.query(patient_representations)
        query = queries[-1:]
        history_values = np.zeros((len(input) - 1, 379))
        b = torch.zeros(1, 379).to(self.device)
        if len(input) > 1:
            i2_seq = torch.cat(list(reversed(i2_seq[:-1])), dim=1)
            o2, h2 = self.retain[1](i2_seq)
            last_seq_medication = h2
            last_seq_medication_emb = self.med_embedding(last_seq_medication)
            encoded_medication = self.medication_encoder(last_seq_medication_emb, src_mask=None)
            prob_g = prob_g.squeeze(dim=1)
            history_keys = query
            b = prob_g * encoded_medication.squeeze(dim=1)
            print("b shape2:", b.shape)
            b = b.view(1, -1)
            print("b shape3:", b.shape)
            b = self.tran(b)
            print("b shape4:", b.shape)
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[1]] = 1
                history_items = torch.mm(torch.Tensor(history_values).to(self.device), eh)
            history_values = torch.FloatTensor(history_values).to(self.device)
        fun2 = torch.mm(query, eh.t())
        key_weights1 = F.softmax(torch.mm(query, eh.t()), dim=-1)
        fact1 = torch.mm(key_weights1, eh)
        w1 = torch.sigmoid(fun2)
        if len(input) > 1:
            history_keys = queries[:(queries.size(0) - 1)]
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1)
            weighted_values = visit_weight.mm(history_values)

            fact2 = torch.mm(weighted_values, eh)
        else:
            fact2 = fact1
        output = self.output(torch.cat([query, fact1], dim=-1))
        a = w1 * (0.9 * output) + 1.85 * b
        ehr_graph = self.sample(eh)
        ddi_graph = self.sample2(x_hddi)
        fin_e = ehr_graph.t()
        fin_d = ddi_graph.t()
        return a, fin_e, fin_d
class HGN_PGS(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, batchSize):
        super(HGN_PGS, self).__init__()
        self.batchSize = batchSize
        self.SH_embedding = torch.nn.Embedding(sh_num, embedding_dim)
        self.SS_embedding = torch.nn.Embedding(ss_num,embedding_dim)
        self.HH_embedding = torch.nn.Embedding(sh_num,embedding_dim)

        self.HGN_PGSSS_Embedding=NCModel(n_nodes=ss_num,output_dim=embedding_dim)
        self.HGN_PGSSH_Embedding = NCModel(n_nodes=sh_num,output_dim=embedding_dim)
        self.HGN_PGSSH_Embedding2 = NCModel(n_nodes=sh_num, output_dim=embedding_dim)
        self.HGN_PGSHH_Embedding=NCModel(n_nodes=hh_num,output_dim=embedding_dim)
        self.sh_ems=torch.nn.Linear(32,32)
        self.SH_mlp_1 = torch.nn.Linear(32, 32)

        self.SH_bn_1_h = torch.nn.BatchNorm1d(32)
        self.SH_tanh_1_h = torch.nn.Tanh()

        self.concatsLinear = torch.nn.Linear(32*2,32)
        self.concathLinear = torch.nn.Linear(32*2,32)

        self.L = torch.nn.Linear(32+27,32)
        self.mlp = torch.nn.Linear(32, 32)
        self.SI_bn = torch.nn.BatchNorm1d(32)
        self.relu = torch.nn.ReLU()


    def forward(self, x_SH, Spare_SH,x_SS,Spare_SS,x_HH,Spare_HH,prescription, kgOneHot):
        x_SH1 = self.SH_embedding(x_SH.long())
        x_SH2 = self.HGN_PGSSH_Embedding.encode(x_SH1.squeeze(), Spare_SH)
        x_SH2 = self.HGN_PGSSH_Embedding.compute_metrics(x_SH2,Spare_SH)
        x_SH4 = self.HGN_PGSSH_Embedding2.encode(x_SH2,Spare_SH)
        x_SH5 = self.HGN_PGSSH_Embedding2.compute_metrics(x_SH4,Spare_SH)
        x_SH3 = self.sh_ems(x_SH1.squeeze())
        x_SH2 = (x_SH2+x_SH3+x_SH5)/3.0
        x_SH2 = self.SH_mlp_1(x_SH2)
        x_SH2 = self.SH_bn_1_h(x_SH2)
        x_SH2 = self.SH_tanh_1_h(x_SH2)

        x_ss1 = self.SS_embedding(x_SS.long())
        x_ss2 = self.HGN_PGSSS_Embedding.encode(x_ss1.squeeze(), Spare_SS)
        x_ss2 = self.HGN_PGSSS_Embedding.compute_metrics(x_ss2, Spare_SS)

        x_hh1 = self.HH_embedding(x_HH.long())
        x_hh2 = self.HGN_PGSHH_Embedding.encode(x_hh1.squeeze(),Spare_HH)
        x_hh2 = self.HGN_PGSHH_Embedding.compute_metrics(x_hh2,Spare_HH)
        x_hh2 = torch.cat((x_hh2.float(), kgOneHot), dim=-1)
        x_hh2 = self.L(x_hh2)
        es = torch.cat([x_SH2[:390], x_ss2], dim=1)
        es = self.concatsLinear(es)
        eh = torch.cat([x_SH2[390:],x_hh2],dim=1)
        eh = self.concathLinear(eh)
        es = es.view(390, -1)
        e_synd = torch.mm(prescription, es)
        preSum = prescription.sum(dim=1).view(-1, 1)
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)
        e_synd_norm = e_synd_norm.view(-1, 32)
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)
        eh = eh.view(805, -1)
        pre = torch.mm(e_synd_norm, eh.t())
        return pre

