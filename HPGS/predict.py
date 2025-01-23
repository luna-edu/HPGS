import argparse
import os
import sys
import time
from collections import defaultdict
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

MODEL_NAME = 'MUATHR'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', default=False,
                        help="no train")
    parser.add_argument('--symptoms', type=str, default='1,5,20',
                        help="symptoms id")
    parser.add_argument('--model_path', type=str, default='saved/HM_PGS/final.model',
                        help="model path")
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.000009, help='Learning rate.')
    parser.add_argument('--device', type=str, default='cuda', help='Running device.')
    args = parser.parse_args()
    return args

def load_graph_data(devices='cuda'):
    sh_edge = np.load('./data/sh_graph.npy')
    sh_edge = sh_edge.tolist()
    sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
    sh_x = torch.tensor([[i] for i in range(2322)], dtype=torch.float)
    sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous()).to(device=devices)

    sh_data_row, sh_data_col = sh_data.edge_index[0], sh_data.edge_index[1]
    sh_data_values = torch.ones(sh_data_row.size(0)).to(devices)
    sh_data_size = (2322, 2322)
    sh_data_s = torch.sparse_coo_tensor(
        indices=torch.stack([sh_data_row, sh_data_col]),
        values=sh_data_values,
        size=sh_data_size
    ).coalesce().to(devices)

    ss_edge = np.load('./data/ss_graph.npy')
    ss_edge = ss_edge.tolist()
    ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
    ss_x = torch.tensor([[i] for i in range(1366)], dtype=torch.float)
    ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous()).to(devices)

    ss_data_row, ss_data_col = ss_data.edge_index[0], ss_data.edge_index[1]
    ss_data_values = torch.ones(ss_data_row.size(0)).to(devices)
    ss_data_size = (1366, 1366)
    ss_data_s = torch.sparse_coo_tensor(
        indices=torch.stack([ss_data_row, ss_data_col]),
        values=ss_data_values,
        size=ss_data_size
    ).coalesce().to(devices)

    hh_edge = np.load('./data/hh_graph.npy').tolist()
    hh_edge_index = torch.tensor(hh_edge, dtype=torch.long)
    hh_x = torch.tensor([[i] for i in range(1366, 2322)], dtype=torch.float)
    hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous()).to(devices)

    hh_data_row, hh_data_col = hh_data.edge_index[0], hh_data.edge_index[1]
    hh_data_values = torch.ones(hh_data_row.size(0)).to(devices)
    hh_data_size = (956, 956)
    hh_data_s = torch.sparse_coo_tensor(
        indices=torch.stack([hh_data_row, hh_data_col]),
        values=hh_data_values,
        size=hh_data_size
    ).coalesce().to(devices)

    hd_edge = np.load('./data/ddi_graph.npy').tolist()
    hd_edge_index = torch.tensor(hd_edge,dtype=torch.long)
    hd_x = torch.tensor([[i] for i in range(1366, 2322)], dtype=torch.float)
    hd_data = Data(x=hd_x, edge_index=hd_edge_index.t().contiguous()).to(devices)

    hd_data_row, hd_data_col = hd_data.edge_index[0], hd_data.edge_index[1]
    hd_data_values = torch.ones(hd_data_row.size(0)).to(devices)
    hd_data_size = (956,956)
    hd_data_s = torch.sparse_coo_tensor(
        indices=torch.stack([hd_data_row, hd_data_col]),
        values=hd_data_values,
        size=hd_data_size
    ).coalesce().to(devices)

    return sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s

def predict_symptoms_to_herbs(
    symptom_ids,
    model,
    sh_data, sh_data_s,
    ss_data, ss_data_s,
    hh_data, hh_data_s,
    hd_data, hd_data_s,
    device='cuda',
    topk=8
):
    model.eval()
    seq_input = [(symptom_ids, [])]

    with torch.no_grad():
        outputs, a, b = model(
            sh_data.x,  sh_data_s,
            ss_data.x,  ss_data_s,
            hh_data.x,  hh_data_s,
            hd_data_s,  seq_input
        )
    preds = torch.sigmoid(outputs).squeeze(0)
    topk_scores, topk_indices = torch.topk(preds, k=topk)
    predicted_herbs = topk_indices.cpu().numpy().tolist()
    return predicted_herbs


def main_train(args):
    devices = args.device
    device = torch.device(args.device)

    sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s = load_graph_data(devices=devices)
    voc_size=(1366,956)
    data = dill.load(open('./data/prescript.pkl','rb'))
    model_name = args.model_name

    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    p_list = [x for x in range(len(data))]
    train_data, dev_test_data = train_test_split(p_list, test_size=0.2, shuffle=True)
    dev_data, test_data = train_test_split(dev_test_data, test_size=0.5, shuffle=False)

    train_dataset = [data[idx] for idx in train_data]
    dev_dataset   = [data[idx] for idx in dev_data]
    test_dataset  = [data[idx] for idx in test_data]


    EPOCH = args.num_epoch
    LR = args.lr

    model = HM_PGS(1366, 956, 2322, 16, 1).to(devices)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.8)
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
                            (1 + -1 * target).float() *
                            F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
            return losses.mean() if size_average else losses.sum()

    contra_loss = ContrastiveLoss(1.5, device=device)

    best_dev_loss = float('inf')
    model_name = args.model_name
    total_train_time = 0.0
    total_val_time = 0.0
    total_test_time = 0.0

    train_start_time = time.time()
    for epoch in range(EPOCH):
        loss_record=[]
        model.train()
        running_loss = 0
        for step, patient_visits in enumerate(train_dataset):
            for idx, adm in enumerate(patient_visits):
                seq_input = patient_visits[:idx+1]
                loss_li_target = np.zeros((1, voc_size[1]))
                loss_li_target[:, adm[1]] = 1

                outputs, a, b = model(
                    sh_data.x, sh_data_s,
                    ss_data.x, ss_data_s,
                    hh_data.x, hh_data_s,
                    hd_data_s,
                    seq_input
                )

                loss_bce = criterion(outputs, torch.FloatTensor(loss_li_target).to(device))
                loss_contra = contra_loss(a, b, torch.FloatTensor(loss_li_target))
                loss_total = 1.1 * loss_bce + 0.1 * loss_contra

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                running_loss += loss_total.item()
                loss_record.append(loss_total.item())

        scheduler.step()
        print('[Epoch {}] train_loss: {:.4f}'.format(epoch + 1, running_loss / len(train_dataset)))

        model.eval()
        dev_loss = 0
        ADMS_dev=[]
        val_start_time = time.time()
        for step, patient_visits in enumerate(dev_dataset):
            for idx, adm in enumerate(patient_visits):
                seq_input = patient_visits[:idx+1]
                loss_li_target = np.zeros((1,voc_size[1]))
                loss_li_target[:, adm[1]] = 1

                with torch.no_grad():
                    outputs,a,b = model(
                        sh_data.x, sh_data_s,
                        ss_data.x, ss_data_s,
                        hh_data.x, hh_data_s,
                        hd_data_s,
                        seq_input
                    )
                    loss_bce = criterion(outputs, torch.FloatTensor(loss_li_target).to(device))
                    loss_contra = contra_loss(a,b,torch.FloatTensor(loss_li_target))
                    loss_total = 0.9*loss_bce + 0.1*loss_contra
                dev_loss += loss_total.item()

        val_end_time = time.time()
        total_val_time += (val_end_time - val_start_time)
        avg_dev_loss = dev_loss / len(dev_dataset)
        print('[Epoch {}] dev_loss: {:.4f}'.format(epoch + 1, avg_dev_loss))

        torch.save(model.state_dict(), os.path.join('saved', model_name, f'epoch_{epoch + 1}.pt'))
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), os.path.join('saved', model_name, 'best_model.pt'))
            print(f"Best model saved at epoch {epoch + 1} with dev_loss: {avg_dev_loss:.4f}")

    torch.save(model.state_dict(), os.path.join('saved', model_name, 'final.model'))
    print("Final model saved.")
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time

    model.eval()
    test_loss = 0
    test_p5 = 0
    test_p10 = 0
    test_p20 = 0
    test_r5 = 0
    test_r10 = 0
    test_r20 = 0

    test_start_time = time.time()
    for step, patient_visits in enumerate(test_dataset):
        for idx, adm in enumerate(patient_visits):
            seq_input = patient_visits[:idx + 1]
            loss_li_target = np.zeros((1, voc_size[1]))
            loss_li_target[:, adm[1]] = 1

            with torch.no_grad():
                outputs, a, b = model(
                    sh_data.x, sh_data_s,
                    ss_data.x, ss_data_s,
                    hh_data.x, hh_data_s,
                    hd_data_s,
                    seq_input
                )
                loss_bce = criterion(outputs, torch.FloatTensor(loss_li_target).to(device))
                loss_contra = contra_loss(a,b,torch.FloatTensor(loss_li_target))
                loss_total = 0.9 * loss_bce + 0.1 * loss_contra
            test_loss += loss_total.item()

            top5 = torch.topk(outputs, 5)[1]
            count5 = sum(int(m in adm[1]) for m in top5[0])
            test_p5 += count5 / 5
            test_r5 += count5 / len(adm[1]) if len(adm[1])>0 else 0

            top10 = torch.topk(outputs, 10)[1]
            count10 = sum(int(m in adm[1]) for m in top10[0])
            test_p10 += count10 / 10
            test_r10 += count10 / len(adm[1]) if len(adm[1])>0 else 0

            top20 = torch.topk(outputs, 20)[1]
            count20 = sum(int(m in adm[1]) for m in top20[0])
            test_p20 += count20 / 20
            test_r20 += count20 / len(adm[1]) if len(adm[1])>0 else 0

    test_end_time = time.time()
    total_test_time += (test_end_time - test_start_time)
    test_size = sum(len(x) for x in test_dataset)
    avg_test_loss = test_loss / len(test_dataset)
    print('test_loss: ', avg_test_loss)
    print('p@5-10-20:', test_p5/test_size, test_p10/test_size, test_p20/test_size)
    print('r@5-10-20:', test_r5/test_size, test_r10/test_size, test_r20/test_size)

    f1_5  = 2*(test_p5/test_size)*(test_r5/test_size)/( (test_p5/test_size)+(test_r5/test_size) ) if (test_p5+test_r5)>0 else 0
    f1_10 = 2*(test_p10/test_size)*(test_r10/test_size)/( (test_p10/test_size)+(test_r10/test_size) ) if (test_p10+test_r10)>0 else 0
    f1_20 = 2*(test_p20/test_size)*(test_r20/test_size)/( (test_p20/test_size)+(test_r20/test_size) ) if (test_p20+test_r20)>0 else 0

    print('f1_5-10-20: ', f1_5, f1_10, f1_20)
    print('eval time：', total_val_time)
    print('train time：', total_train_time)
    print('test time：', total_test_time)

def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            mapping[idx] = line.strip()
    return mapping

def map_ids_to_names(ids, mapping):
    return [mapping.get(i, f"unknown ID({i})") for i in ids]
def parse_test_dataset(raw_dataset):
    parsed_dataset = []
    for case in raw_dataset:
        for visit in case:
            if len(visit) == 2:
                symptom_ids, herb_ids = visit
                parsed_dataset.append((symptom_ids, herb_ids))
    return parsed_dataset
def main():
    args = get_args()
    symptoms_mapping = load_mapping('./data/sys_index.txt')
    herbs_mapping = load_mapping('./data/herbs_index.txt')
    if args.predict:
        if not args.symptoms:
            print("use --symptoms example: --symptoms '1,5,20'")
            return
        symptom_ids = [int(x) for x in args.symptoms.split(',')]
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        sh_data, sh_data_s, ss_data, ss_data_s, hh_data, hh_data_s, hd_data, hd_data_s = load_graph_data(devices=device)
        model = HM_PGS(1366, 956, 2322, 16, 1).to(device)
        print(f"Loading model from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        predicted_herbs = predict_symptoms_to_herbs(
            symptom_ids,
            model,
            sh_data, sh_data_s,
            ss_data, ss_data_s,
            hh_data, hh_data_s,
            hd_data, hd_data_s,
            device=device,
            topk=8
        )
        symptom_names = map_ids_to_names(symptom_ids, symptoms_mapping)
        herb_names = map_ids_to_names(predicted_herbs, herbs_mapping)
        voc_size = (1366, 956)
        data = dill.load(open('./data/prescript.pkl', 'rb'))
        model_name = args.model_name
        if not os.path.exists(os.path.join("saved", model_name)):
            os.makedirs(os.path.join("saved", model_name))

        p_list = [x for x in range(len(data))]
        test_dataset = [data[idx] for idx in p_list]
        parsed_test_dataset = parse_test_dataset(test_dataset)
        case_study(
            parsed_test_dataset,
            model,
            sh_data, sh_data_s,
            ss_data, ss_data_s,
            hh_data, hh_data_s,
            hd_data, hd_data_s,
            symptoms_mapping,
            herbs_mapping,
            device='cuda',
            topk=8
        )
    else:
        main_train(args)
def case_study(
    test_dataset,
    model,
    sh_data, sh_data_s,
    ss_data, ss_data_s,
    hh_data, hh_data_s,
    hd_data, hd_data_s,
    symptoms_mapping,
    herbs_mapping,
    device='cuda',
    topk=8,
    output_excel="results.xlsx",
    precision_threshold=0.5,
    recall_threshold=0.5
):
    results = []
    correct_cases = []
    partial_cases = []
    error_cases = []

    for case in test_dataset:
        symptom_ids, true_herb_ids = case
        predicted_herb_ids = predict_symptoms_to_herbs(
            symptom_ids,
            model,
            sh_data, sh_data_s,
            ss_data, ss_data_s,
            hh_data, hh_data_s,
            hd_data, hd_data_s,
            device=device,
            topk=topk
        )
        symptom_names = map_ids_to_names(symptom_ids, symptoms_mapping)
        true_herbs = map_ids_to_names(true_herb_ids, herbs_mapping)
        predicted_herbs = map_ids_to_names(predicted_herb_ids, herbs_mapping)
        true_set = set(true_herb_ids)
        pred_set = set(predicted_herb_ids)
        intersection = true_set.intersection(pred_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(true_set) if true_set else 0
        if pred_set == true_set:
            comment = "all correct"
            correct_cases.append((symptom_names, true_herbs, predicted_herbs, comment))
        elif precision >= precision_threshold or recall >= recall_threshold:
            comment = f" correct (Precision: {precision:.2f}, Recall: {recall:.2f})"
            partial_cases.append((symptom_names, true_herbs, predicted_herbs, comment))
        else:
            comment = "error"
            error_cases.append((symptom_names, true_herbs, predicted_herbs, comment))
        results.append({
            "input": ", ".join(symptom_names),
            "label": ", ".join(true_herbs),
            "output": ", ".join(predicted_herbs),
            "Precision": precision,
            "Recall": recall,
            "comment": comment
        })
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"save result to {output_excel}")
    def print_cases(cases, title):
        print(f"\n{title}")
        print("-" * 90)
        for symptom_names, true_herbs, predicted_herbs, comment in cases:
            print(f"{', '.join(symptom_names):<25} | {', '.join(true_herbs):<25} | {', '.join(predicted_herbs):<25} | {comment:<15}")
    return correct_cases, partial_cases, error_cases

if __name__=='__main__':
    main()
