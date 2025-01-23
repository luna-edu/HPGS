import sys
import os
import parameter
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from HPGS_models import HGN_PGS, NCModel
from pytorchtools import EarlyStopping
import time
from sklearn.metrics import roc_auc_score
import types
from torch_sparse import SparseTensor
from util import load_obj, presDataset

MODEL_NAME = 'HGN_PGS'
device = 'cuda'
import warnings
para = parameter.para(lr=3e-3,rec=6e-5,drop=0.1,batchSize=32,epoch=50,dev_ratio=0.2,test_ratio=0.2)

warnings.filterwarnings("ignore", message="where received a uint8 condition tensor", category=UserWarning)
warnings.filterwarnings("ignore", message="nn.functional.sigmoid is deprecated")


sh_edge = np.load('./data/sh_graph.npy')
sh_edge = sh_edge.tolist()
sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
sh_x = torch.tensor([[i] for i in range(1195)], dtype=torch.float)
sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous()).to(device)
sh_data_row,sh_data_col=sh_data.edge_index[0],sh_data.edge_index[1]
sh_data_values = torch.ones(79870).to(device)
sh_data_size = (1195,1195)
sh_data__nnz=79870
sh_data_sparse_tensor = torch.sparse_coo_tensor(indices=torch.stack([sh_data_row,sh_data_col]), values=sh_data_values, size=sh_data_size).to(device)

ss_edge = np.load('./data/ss_graph.npy')
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous()).to(device)
ss_data_row,ss_data_col =ss_data.edge_index[0], ss_data.edge_index[1]
ss_data_size = (390,390)
ss_data_nnz =2546
ss_data_values = torch.ones(2546).to(device)
ss_data_sparse_tensor=torch.sparse_coo_tensor(indices=torch.stack([ss_data_row,ss_data_col]), values=ss_data_values, size=ss_data_size).to(device)

hh_edge = np.load('./data/hh_graph.npy').tolist()
hh_edge_index = torch.tensor(hh_edge, dtype=torch.long) - 390
hh_x = torch.tensor([[i] for i in range(390, 1195)], dtype=torch.float)
hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous()).to(device)
hh_data_row=hh_data.edge_index[0]
hh_data_col=hh_data.edge_index[1]
hh_data_size=(805,805)
hh_data_values=torch.ones(9038).to(device)
hh_data_spare_tensor=torch.sparse_coo_tensor(indices=torch.stack([hh_data_row,hh_data_col]),values=hh_data_values,size=hh_data_size).to(device)

prescript = pd.read_csv('./data/prescript.csv', encoding='utf-8')
pLen = len(prescript)
pS_list = [[0]*390 for _ in range(pLen)]
pS_array = np.array(pS_list)

pH_list = [[0] * 805 for _ in range(pLen)]
pH_array = np.array(pH_list)

for i in range(pLen):
    j = eval(prescript.iloc[i, 0])
    pS_array[i, j] = 1

    k = eval(prescript.iloc[i, 1])
    k = [x - 390 for x in k]
    pH_array[i, k] = 1

herbCount = load_obj('./data/herbID')
herbCount = np.array(list(herbCount.values()))

kg_oneHot = np.load('./data/herb_oneHot.npy')
kg_oneHot = torch.from_numpy(kg_oneHot).float().to(device)

p_list = [x for x in range(pLen)]
x_train, x_dev_test = train_test_split(p_list, test_size= (para.dev_ratio + para.test_ratio), shuffle=True,
                                       random_state=2020)

x_dev, x_test = train_test_split(x_dev_test, test_size=1 - 0.5, shuffle=False, random_state=2023)
print("train_size: ", len(x_train), "dev_size: ", len(x_dev), "test_size: ", len(x_test))


train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev])
test_dataset = presDataset(pS_array[x_test], pH_array[x_test])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=para.batchSize)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=para.batchSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=para.batchSize)
models = HGN_PGS(390,805,1195,32,1).to(device)
criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
optimizer = torch.optim.Adam(models.parameters(), lr=para.lr, weight_decay=para.rec)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5 , gamma=0.4)
early_stopping = EarlyStopping(patience=10, verbose=True)
print('device: ', device)
epsilon = 1e-13
total_train_time = 0.0
total_val_time = 0.0
total_test_time = 0.0
for epoch in range(para.epoch):
    models.to(device)
    models.train()
    running_loss = 0.0
    train_start_time = time.time()
    for i, (sid, hid) in enumerate(train_loader):
        sid, hid = sid.to(device), hid.to(device)
        sid, hid = sid.float(), hid.float()
        optimizer.zero_grad()
        outputs = models(sh_data.x, sh_data_sparse_tensor, ss_data.x, ss_data_sparse_tensor,hh_data.x, hh_data_spare_tensor, sid, kg_oneHot)
        loss = criterion(outputs, hid)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
    train_end_time = time.time()
    epoch_train_time = train_end_time - train_start_time
    total_train_time += epoch_train_time
    print('[Epoch {}]train_loss: {:.4f} | Train Time: {:.2f} sec'.format(
        epoch + 1, running_loss / len(train_loader), epoch_train_time))
    models.eval()
    dev_loss = 0
    val_start_time = time.time()
    for tsid, thid in dev_loader:
        tsid, thid = tsid.float(), thid.float()
        tsid,thid = tsid.to(device),thid.to(device)
        outputs = models(sh_data.x, sh_data_sparse_tensor, ss_data.x, ss_data_sparse_tensor,hh_data.x, hh_data_spare_tensor, tsid, kg_oneHot)
        dev_loss += criterion(outputs, thid).item()

    scheduler.step()
    val_end_time = time.time()
    epoch_val_time = val_end_time - val_start_time
    total_val_time += epoch_val_time
    print('[Epoch {}]dev_loss: {:.4f} | Validation Time: {:.2f} sec'.format(
        epoch + 1, dev_loss / len(dev_loader), epoch_val_time))
    print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(dev_loader))
    if epoch>para.epoch-10:
        early_stopping(dev_loss / len(dev_loader), models)
        if early_stopping.early_stop:
            print("Early stopping")
            break

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

models.load_state_dict(torch.load('checkpoint.pt'))
torch.save(models.state_dict(), 'model_parameters.pth')
models.eval()
test_loss = 0

test_p5 = 0
test_p10 = 0
test_p20 = 0

test_r5 = 0
test_r10 = 0
test_r20 = 0

test_f1_5 = 0
test_f1_10 = 0
test_f1_20 = 0

test_start_time = time.time()
for tsid, thid in test_loader:
    tsid, thid = tsid.float(), thid.float()
    tsid,thid = tsid.to(device),thid.to(device)
    outputs = models(sh_data.x, sh_data_sparse_tensor, ss_data.x, ss_data_sparse_tensor, hh_data.x,
                     hh_data_spare_tensor, tsid, kg_oneHot)
    test_loss += criterion(outputs, thid).item()
    for i, hid in enumerate(thid):
        trueLabel = []
        for idx, val in enumerate(hid):
            if val == 1:
                trueLabel.append(idx)
        top5 = torch.topk(outputs[i], 5)[1]
        count = 0
        for m in top5:
            if m in trueLabel:
                count += 1
        test_p5 += count / 5
        test_r5 += count / len(trueLabel)
        top10 = torch.topk(outputs[i], 10)[1]
        count = 0
        for m in top10:
            if m in trueLabel:
                count += 1
        test_p10 += count / 10
        test_r10 += count / len(trueLabel)
        top20 = torch.topk(outputs[i], 20)[1]
        count = 0
        for m in top20:
            if m in trueLabel:
                count += 1
        test_p20 += count / 20
        test_r20 += count / len(trueLabel)
test_end_time = time.time()
total_test_time =test_end_time -test_start_time
print('test_loss: ', test_loss / len(test_loader))
print('p5-10-20:', test_p5 / len(x_test), test_p10 / len(x_test), test_p20 / len(x_test))
print('r5-10-20:', test_r5 / len(x_test), test_r10 / len(x_test), test_r20 / len(x_test))
print('f1_5-10-20: ',
      2 * (test_p5 / len(x_test)) * (test_r5 / len(x_test)) / ((test_p5 / len(x_test)) + (test_r5 / len(x_test))),
      2 * (test_p10 / len(x_test)) * (test_r10 / len(x_test)) / ((test_p10 / len(x_test)) + (test_r10 / len(x_test))),
      2 * (test_p20 / len(x_test)) * (test_r20 / len(x_test)) / ((test_p20 / len(x_test)) + (test_r20 / len(x_test))))
