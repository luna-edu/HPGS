import argparse
import os
import sys
import time
from collections import defaultdict
import itertools
import pandas as pd
import dill
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from HPGS_models import HM_PGS
from model import HM_PGS
from util import presDataset, llprint, multi_label_metric
import torch.nn.functional as F

MODEL_NAME = 'HM_PGS'

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def print_to_log(string, log_fp):
    llprint(string)
    print(string, file=log_fp)

GRID_PARAMS = {
    "lr": [0.000009],
    "num_epoch": [100],
    "dropout_rate": [0.1],
    "device": ["cuda"],
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/', help='set the data path')
    args = parser.parse_args()
    return args

def get_grid_combinations(grid_params):
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def initialize_data(devices):
    sh_edge = np.load('./data/sh_graph.npy')
    sh_edge = sh_edge.tolist()
    sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
    sh_x = torch.tensor([[i] for i in range(1353)], dtype=torch.float)
    sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous()).to(device=devices)
    sh_data_adj = SparseTensor(row=sh_data.edge_index[0], col=sh_data.edge_index[1],
                               sparse_sizes=(1353, 1353))
    sh_data_row, sh_data_col = sh_data.edge_index[0], sh_data.edge_index[1]
    sh_data_values = torch.ones(257).to(devices)
    sh_data_size = (1353, 1353)
    sh_data_s = torch.sparse_coo_tensor(indices=torch.stack([sh_data_row, sh_data_col]), values=sh_data_values, size=sh_data_size).to(devices)
    ss_edge = np.load('./data/ss_graph.npy')
    ss_edge = ss_edge.tolist()
    ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
    ss_x = torch.tensor([[i] for i in range(974)], dtype=torch.float)
    ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous()).to(device=devices)
    ss_data_adj = SparseTensor(row=ss_data.edge_index[0], col=ss_data.edge_index[1],
                               sparse_sizes=(974, 974))
    ss_data_row, ss_data_col = ss_data.edge_index[0], ss_data.edge_index[1]
    ss_data_values = torch.ones(15455).to(devices)
    ss_data_size = (974,974)
    ss_data_s = torch.sparse_coo_tensor(indices=torch.stack([ss_data_row, ss_data_col]), values=ss_data_values,
                                        size=ss_data_size).to(devices)
    hh_edge = np.load('./data/hh_graph.npy').tolist()
    hh_edge_index = torch.tensor(hh_edge, dtype=torch.long)
    hh_x = torch.tensor([[i] for i in range(974,1353)], dtype=torch.float)
    hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous()).to(device=devices)
    hh_data_adj = SparseTensor(row=hh_data.edge_index[0], col=hh_data.edge_index[1],
                               sparse_sizes=(379, 379))
    hh_data_row, hh_data_col = hh_data.edge_index[0], hh_data.edge_index[1]
    hh_data_values = torch.ones(10487).to(devices)
    hh_data_size = (379,379)
    hh_data_s = torch.sparse_coo_tensor(indices=torch.stack([hh_data_row, hh_data_col]), values=hh_data_values,
                                        size=hh_data_size).to(devices)
    hd_edge = np.load('./data/ddi_graph.npy').tolist()
    hd_edge_index = torch.tensor(hd_edge,dtype=torch.long)
    hd_x = torch.tensor([[i] for i in range(973, 1353)], dtype=torch.float)
    hd_data = Data(x=hd_x, edge_index=hd_edge_index.t().contiguous()).to(device=devices)
    hd_data_row, hd_data_col = hd_data.edge_index[0], hd_data.edge_index[1]
    hd_data_values = torch.ones(1).to(devices)
    hd_data_size = (379,379)
    hd_data_s = torch.sparse_coo_tensor(indices=torch.stack([hd_data_row, hd_data_col]), values=hd_data_values,
                                        size=hd_data_size).to(devices)
    return sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s

def train_model(model, train_dataset, criterion, optimizer, contra_loss, device, voc_size, log_fp, sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s):
    model.train()
    running_loss = 0
    loss_record = []
    for step, input in enumerate(train_dataset):
        for idx, adm in enumerate(input):
            seq_input = input[:idx+1]
            loss_li_target = np.zeros((1, voc_size[1]))
            loss_li_target[:, adm[1]] = 1
            outputs, a, b = model(sh_data.x, sh_data_s, ss_data.x, ss_data_s, hh_data.x, hh_data_s, hd_data_s, seq_input)
            loss = criterion(outputs, torch.FloatTensor(loss_li_target).to(device))
            loss1 = contra_loss(a, b, torch.FloatTensor(loss_li_target).to(device))
            loss = 1.2 * loss + 0.1 * loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            loss_record.append(loss.item())
    return running_loss / len(train_dataset), loss_record

def eval_model(model, data_eval, voc_size, device, log_fp, epoch, args, sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s, criterion, contra_loss):
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [], [], [], [], []
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    with torch.no_grad():
        for step, input in enumerate(data_eval):
            y_gt = []
            y_pred = []
            y_pred_prob = []
            y_pred_label = []
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                loss_li_target = np.zeros((1, voc_size[1]))
                loss_li_target[:, adm[1]] = 1
                outputs, _, _ = model(sh_data.x, sh_data_s, ss_data.x, ss_data_s, hh_data.x, hh_data_s, hd_data_s, seq_input)
                y_gt_tmp = np.zeros(voc_size[1])
                y_gt_tmp[adm[1]] = 1
                y_gt.append(y_gt_tmp)
                target_output1 = torch.sigmoid(outputs).cpu().numpy()[0]
                y_pred_prob.append(target_output1)
                y_pred_tmp = target_output1.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)
            smm_record.append(y_pred_label)
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
            case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step + 1, len(data_eval)))
    avg_ja = np.mean(ja)
    avg_prauc = np.mean(prauc)
    avg_avg_p = np.mean(avg_p)
    avg_avg_r = np.mean(avg_r)
    avg_avg_f1 = np.mean(avg_f1)
    llprint('\tJaccard: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        avg_ja, avg_prauc, avg_avg_p, avg_avg_r, avg_avg_f1
    ))
    print_to_log('\tJaccard: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        avg_ja, avg_prauc, avg_avg_p, avg_avg_r, avg_avg_f1
    ), log_fp)
    dill.dump(case_study, open(os.path.join('saved', args["model_name"], f'case_study_epoch_{epoch +1}.pkl'), 'wb'))
    return avg_ja, avg_prauc, avg_avg_p, avg_avg_r, avg_avg_f1
def test_model(model, test_dataset, voc_size, device, log_fp, sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s, criterion, contra_loss):
    model.eval()
    test_loss = 0
    test_p5 = 0
    test_p10 = 0
    test_p20 = 0
    test_r5 = 0
    test_r10 = 0
    test_r20 = 0
    ADMS_T=[]
    true_lable_test=[]
    test_start_time = time.time()
    with torch.no_grad():
        for step, input in enumerate(test_dataset):
            for idx, adm in enumerate(input):
                ADMS_T.append(adm[1])
            for adms in ADMS_T:
                for a in adms:
                    true_lable_test.append(a)
            for idx, adm in enumerate(input):
                seq_input = input[:idx + 1]
                loss_li_target = np.zeros((1, voc_size[1]))
                loss_li_target[:, adm[1]] = 1
                outputs, a, b = model(sh_data.x, sh_data_s, ss_data.x, ss_data_s, hh_data.x, hh_data_s, hd_data_s, seq_input)
                loss = criterion(outputs, torch.FloatTensor(loss_li_target).to(device))
                loss1 = contra_loss(a, b, torch.FloatTensor(loss_li_target).to(device))
                loss = 0.9 * loss + 0.1 * loss1
                test_loss += loss.item()
                top5 = torch.topk(outputs, 5)[1]
                count = 0
                for m in top5[0]:
                    if m in adm[1]:
                        count += 1
                test_p5 += count / 5
                test_r5 += count / len(adm[1])
                top10 = torch.topk(outputs, 10)[1]
                count = 0
                for m in top10[0]:
                    if m in adm[1]:
                        count += 1
                test_p10 += count / 10
                test_r10 += count / len(adm[1])
                top20 = torch.topk(outputs, 20)[1]
                count = 0
                for m in top20[0]:
                    if m in adm[1]:
                        count += 1
                test_p20 += count / 20
                test_r20 += count / len(adm[1])
    test_end_time = time.time()
    total_test_time = test_end_time - test_start_time
    print_to_log("--------------------------------------------------------------------------------------------------------\n", log_fp)
    print_to_log(f'test_loss: {test_loss / len(test_dataset):.4f}\n', log_fp)
    print_to_log(f'p5-10-20: {test_p5 / len(test_dataset):.4f}, {test_p10 / len(test_dataset):.4f}, {test_p20 / len(test_dataset):.4f}\n', log_fp)
    print_to_log(f'r5-10-20: {test_r5 / len(test_dataset):.4f}, {test_r10 / len(test_dataset):.4f}, {test_r20 / len(test_dataset):.4f}\n', log_fp)
    print_to_log(f'f1_5-10-20: '
                f'{2 * (test_p5 / len(test_dataset)) * (test_r5 / len(test_dataset)) / ((test_p5 / len(test_dataset)) + (test_r5 / len(test_dataset)) + 1e-13):.4f}, '
                f'{2 * (test_p10 / len(test_dataset)) * (test_r10 / len(test_dataset)) / ((test_p10 / len(test_dataset)) + (test_r10 / len(test_dataset)) + 1e-13):.4f}, '
                f'{2 * (test_p20 / len(test_dataset)) * (test_r20 / len(test_dataset)) / ((test_p20 / len(test_dataset)) + (test_r20 / len(test_dataset)) + 1e-13):.4f}\n', log_fp)
    print_to_log("===========================================\n", log_fp)
    print_to_log(f'time ：{total_test_time:.2f}s\n', log_fp)
    return test_loss / len(test_dataset), {
        "p5": test_p5 / len(test_dataset),
        "p10": test_p10 / len(test_dataset),
        "p20": test_p20 / len(test_dataset),
        "r5": test_r5 / len(test_dataset),
        "r10": test_r10 / len(test_dataset),
        "r20": test_r20 / len(test_dataset),
        "f1_5": 2 * (test_p5 / len(test_dataset)) * (test_r5 / len(test_dataset)) / ((test_p5 / len(test_dataset)) + (test_r5 / len(test_dataset)) + 1e-13),
        "f1_10": 2 * (test_p10 / len(test_dataset)) * (test_r10 / len(test_dataset)) / ((test_p10 / len(test_dataset)) + (test_r10 / len(test_dataset)) + 1e-13),
        "f1_20": 2 * (test_p20 / len(test_dataset)) * (test_r20 / len(test_dataset)) / ((test_p20 / len(test_dataset)) + (test_r20 / len(test_dataset)) + 1e-13)
    }

def main():
    args = get_args()
    grid_combinations = get_grid_combinations(GRID_PARAMS)
    for config in grid_combinations:
        print(f"\n{'='*20} Running training with config: {config} {'='*20}\n")
        current_args = vars(args).copy()
        current_args.update(config)
        print("Current Arguments:")
        for key, value in current_args.items():
            print(f"  {key}: {value}")
        device = torch.device(current_args["device"])
        sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s = initialize_data(device)
        voc_size = (974,379)
        data = dill.load(open('./data/prescript.pkl','rb'))
        p_list = list(range(len(data)))
        train_data, dev_test_data = train_test_split(p_list, test_size=0.2, shuffle=True)
        dev_data, test_data = train_test_split(dev_test_data, test_size=0.5, shuffle=False)
        train_dataset = [data[index] for index in train_data]
        dev_dataset = [data[index] for index in dev_data]
        test_dataset = [data[index] for index in test_data]
        model_name = current_args["model_name"]
        save_dir = os.path.join("saved", model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        log_filename = f'{timestamp}_lr{current_args["lr"]}_epoch{current_args["num_epoch"]}.log'
        log_path = os.path.join(save_dir, log_filename)
        log_fp = open(log_path, 'w')
        model = HGCNMHR(973,379,1353,16,1).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=current_args["lr"], weight_decay=0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        class ContrastiveLoss(nn.Module):
            def __init__(self, margin, device):
                super(ContrastiveLoss, self).__init__()
                self.margin = margin
                self.eps = 1e-9
                self.device = device

            def forward(self, output1, output2, target, size_average=True):
                target = target.to(self.device)
                distances = (output2 - output1).pow(2).sum(1).to(self.device)
                losses = 0.5 * (target.float() * distances +
                                (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
                return losses.mean() if size_average else losses.sum()

        contra_loss = ContrastiveLoss(1.5, device=device)
        history = defaultdict(list)
        best_ja = 0
        train_start_time = time.time()
        EPOCH = current_args["num_epoch"]
        for epoch in range(EPOCH):
            start_time = time.time()
            running_loss, batch_losses = train_model(model, train_dataset, criterion, optimizer, contra_loss, device, voc_size, log_fp, sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s)
            history['train_loss'].append(running_loss)
            print_to_log(f'[Epoch {epoch + 1}] train_loss: {running_loss:.4f}\n', log_fp)
            scheduler.step()
            ja, prauc, avg_p, avg_r, avg_f1 = eval_model(model, dev_dataset, voc_size, device, log_fp, epoch, current_args, sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s, criterion, contra_loss)
            history['ja'].append(ja)
            history['prauc'].append(prauc)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            model_save_path = os.path.join(save_dir, f'Epoch_{epoch + 1}_JA_{ja:.4f}.model')
            torch.save(model.state_dict(), model_save_path)
            print_to_log(f'Saved model to {model_save_path}\n', log_fp)
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch + 1
                best_ja = ja
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            remaining_epochs = EPOCH - (epoch + 1)
            estimated_left_time = elapsed_time * remaining_epochs
            print_to_log(f'\tEpoch: {epoch + 1}, Loss: {running_loss:.4f}, One Epoch Time: {elapsed_time:.2f}m, Appro Left Time: {estimated_left_time:.2f}m\n', log_fp)
        train_end_time = time.time()
        total_train_time = train_end_time - train_start_time
        history_save_path = os.path.join(save_dir, f'history_{timestamp}.pkl')
        dill.dump(history, open(history_save_path, 'wb'))
        print_to_log(f'Saved training history to {history_save_path}\n', log_fp)
        test_loss, test_metrics = test_model(model, test_dataset, voc_size, device, log_fp, sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s, criterion, contra_loss)
        print_to_log(f'test_loss: {test_loss:.4f}\n', log_fp)
        print_to_log(f'p5-10-20: {test_metrics["p5"]:.4f}, {test_metrics["p10"]:.4f}, {test_metrics["p20"]:.4f}\n', log_fp)
        print_to_log(f'r5-10-20: {test_metrics["r5"]:.4f}, {test_metrics["r10"]:.4f}, {test_metrics["r20"]:.4f}\n', log_fp)
        print_to_log(f'f1_5-10-20: {test_metrics["f1_5"]:.4f}, {test_metrics["f1_10"]:.4f}, {test_metrics["f1_20"]:.4f}\n', log_fp)
        print_to_log(f'time：{total_train_time:.2f}s\n', log_fp)
        log_fp.close()

if __name__ == '__main__':
    main()
