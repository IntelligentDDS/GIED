import logging

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from tqdm import trange,tqdm
from torch_geometric.nn import GATConv, NNConv, global_max_pool, global_mean_pool, global_add_pool, GlobalAttention
import IPython
import matplotlib.pyplot as plt
from torch.nn.init import xavier_normal_, zeros_
from torch_geometric.data import Data, DataLoader
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from .algorithm_utils import Algorithm, PyTorchUtils

def bf_search(labels, scores):
    """
    Find the a good threshold using the training set
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    m = (-1., -1., -1., None)
    m_t = 0.0
    for threshold in sorted(list(scores))[1:-1]:
        target = precision_recall_fscore_support(labels, (scores > threshold).astype('int'), average = 'binary')
        if target[2] > m[2]:
            m_t = threshold
            m = target
    print(m, m_t)
    return m, m_t

class callSpatioDevNet(Algorithm, PyTorchUtils):
    def __init__(self, name: str='SpatioDevNetPackage', num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
                 input_dim: int = None, hidden_dim: int = 20, edge_attr_len: int = 60, global_fea_len: int = 2,
                 num_layers: int = 2, edge_module: str = 'linear', act: bool = True, pooling: str = 'attention', is_bilinear: bool = False,
                 nonlinear_scorer: bool = False, head: int = 4, aggr: str = 'mean', concat: bool = False, dropout: float = 0.4,
                 weight_decay: float = 1e-2, loss_func: str = 'focal_loss', seed: int = None, gpu: int = None, ipython = True, details = True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_len = edge_attr_len
        self.global_fea_len = global_fea_len
        self.num_layers = num_layers

        self.edge_module = edge_module
        self.act = act
        self.pooling = pooling
        self.is_bilinear = is_bilinear
        self.nonlinear_scorer = nonlinear_scorer
        self.head = head
        self.aggr = aggr
        self.concat = concat
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.final_train_fscore = None
        self.ipython = ipython

        self.loss_func = loss_func

        self.devnet = SpatioDevNetModule(self.input_dim, self.hidden_dim, self.edge_attr_len, self.global_fea_len, self.num_layers,
                 self.edge_module, self.act, self.pooling, self.is_bilinear, self.nonlinear_scorer, self.head, self.aggr,
                 self.concat, self.dropout, self.seed, self.gpu)
        #self.sscaler = StandardScaler()
        self.loss_logs = {}

    def fit(self, datalist: list, valid_list: list = None, log_step: int = 20, patience: int = 10, valid_proportion: float = 0.0, early_stop_fscore: float = None):
        #data = self.sscaler.fit_transform(data)
        if valid_list is not None:
            train_list = datalist
            train_loader = DataLoader(dataset=train_list, batch_size=self.batch_size, shuffle=True)
            valid_loader_of_train_data = DataLoader(dataset=train_list, batch_size=len(train_list), shuffle=False)
            valid_loader = DataLoader(dataset=valid_list, batch_size=len(valid_list), shuffle=False)
        elif valid_proportion != 0:
            split_point = int(valid_proportion * len(datalist))
            shuffle_list = copy.deepcopy(datalist)
            random.shuffle(shuffle_list)

            train_list = shuffle_list[:-split_point]
            valid_list = shuffle_list[-split_point:]

            train_loader = DataLoader(dataset=train_list, batch_size=self.batch_size, shuffle=True)
            valid_loader_of_train_data = DataLoader(dataset=train_list, batch_size=len(train_list), shuffle=False)
            valid_loader = DataLoader(dataset=valid_list, batch_size=len(valid_list), shuffle=False)
        else:
            train_list = datalist
            train_loader = DataLoader(dataset=train_list, batch_size=self.batch_size, shuffle=True)
            valid_loader_of_train_data = DataLoader(dataset=train_list, batch_size=len(train_list), shuffle=False)


        self.to_device(self.devnet)
        optimizer = torch.optim.Adam(self.devnet.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        iters_per_epoch = len(train_loader)
        counter = 0
        best_val_fscore = 0
        best_train_fscore = 0

        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            self.devnet.train()
            for (i, batch) in enumerate(tqdm(train_loader)):

                output, feature = self.devnet(batch)
                if self.loss_func == 'focal_loss':
                    total_loss = SpatioDevNetModule.bce_focal_loss_function(output, batch.y)
                elif self.loss_func == 'dev_loss':
                    total_loss = SpatioDevNetModule.deviation_loss_function(output, batch.y, batch.confidence)
                else:
                    total_loss = SpatioDevNetModule.cross_entropy_loss_function(output, batch.y)

                loss = {}
                loss['total_loss'] = total_loss.data.item()

                self.devnet.zero_grad()
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.devnet.parameters(), 5)
                optimizer.step()

                if (i+1) % log_step == 0:
                    if self.ipython:
                        IPython.display.clear_output()
                        plt.figure(figsize=(12, 6))
                    else:
                        plt.figure(figsize=(12, 6))
                    log = "Epoch [{}/{}], Iter [{}/{}]".format(
                        epoch+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    plt_ctr = 1
                    if not self.loss_logs:
                        for loss_key in loss:
                            self.loss_logs[loss_key] = [loss[loss_key]]
                            plt.subplot(2,3,plt_ctr)
                            plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                            plt.legend()
                            plt_ctr += 1
                    else:
                        for loss_key in loss:
                            self.loss_logs[loss_key].append(loss[loss_key])
                            plt.subplot(2,3,plt_ctr)
                            plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                            plt.legend()
                            plt_ctr += 1
                        if 'train_prf' in self.loss_logs:
                            plt.subplot(2, 3, plt_ctr)
                            plt.plot(np.array(self.loss_logs['train_prf'])[:, 0], label='train_precision')
                            plt.plot(np.array(self.loss_logs['train_prf'])[:, 1], label='train_recall')
                            plt.plot(np.array(self.loss_logs['train_prf'])[:, 2], label='train_fscore')
                            plt.legend()
                            plt_ctr += 1
                        if 'valid_loss' in self.loss_logs:
                            for valid_item in ['valid_loss', 'valid_precision', 'valid_recall', 'valid_fscore']:
                                plt.subplot(2,3,plt_ctr)
                                plt.plot(np.array(self.loss_logs[valid_item]), label=valid_item)
                                plt.legend()
                                plt_ctr += 1
                            print("valid_fscore:", self.loss_logs['valid_fscore'])
                    if self.ipython:
                        plt.show()
                    else:
                        plt.savefig("test.png", dpi=120)
            if valid_proportion != 0 or valid_list is not None:
                self.devnet.eval()
                valid_losses = []
                valid_outputs = []
                train_outputs = []
                for (i,batch) in enumerate(tqdm(valid_loader_of_train_data)):
                    output, feature = self.devnet(batch)
                    train_outputs.append(output.data.cpu().numpy())
                train_outputs = np.concatenate(train_outputs)
                for (i,batch) in enumerate(tqdm(valid_loader)):
                    output, feature = self.devnet(batch)
                    if self.loss_func == 'focal_loss':
                        total_loss = SpatioDevNetModule.bce_focal_loss_function(output, batch.y)
                    elif self.loss_func == 'dev_loss':
                        total_loss = SpatioDevNetModule.deviation_loss_function(output, batch.y, batch.confidence)
                    else:
                        total_loss = SpatioDevNetModule.cross_entropy_loss_function(output, batch.y)
                    valid_outputs.append(output.data.cpu().numpy())
                    valid_losses.append(total_loss.item())
                valid_outputs = np.concatenate(valid_outputs)
                valid_loss = np.average(valid_losses)

                train_labels = [int(item.y) for item in train_list]
                valid_labels = [int(item.y) for item in valid_list]
                m, m_t = bf_search(train_labels, train_outputs)

                valid_precision, valid_recall, valid_fscore, _ = precision_recall_fscore_support(valid_labels, (valid_outputs > m_t).astype('int'), average='binary')

                if 'valid_loss' in self.loss_logs:
                    self.loss_logs['train_prf'].append(m[:-1])
                    self.loss_logs['valid_loss'].append(valid_loss)
                    self.loss_logs['valid_precision'].append(valid_precision)
                    self.loss_logs['valid_recall'].append(valid_recall)
                    self.loss_logs['valid_fscore'].append(valid_fscore)
                else:
                    self.loss_logs['train_prf'] = [m[:-1]]
                    self.loss_logs['valid_loss'] = [valid_loss]
                    self.loss_logs['valid_precision'] = [valid_precision]
                    self.loss_logs['valid_recall'] = [valid_recall]
                    self.loss_logs['valid_fscore'] = [valid_fscore]

                if valid_fscore > best_val_fscore:
                    best_val_fscore = valid_fscore
                    self.final_train_fscore = m[2]
                    torch.save(self.devnet.state_dict(), 'checkpoints/' + self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt')
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print ("early stoppong")
                        self.devnet.load_state_dict(torch.load('checkpoints/' + self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt'))
                        break
            elif early_stop_fscore is not None:
                self.devnet.eval()
                train_outputs = []
                for (i, batch) in enumerate(tqdm(valid_loader_of_train_data)):
                    output, feature = self.devnet(batch)
                    train_outputs.append(output.data.cpu().numpy())
                train_outputs = np.concatenate(train_outputs)
                train_labels = [int(item.y) for item in train_list]
                m, m_t = bf_search(train_labels, train_outputs)
                if 'train_prf' in self.loss_logs:
                    self.loss_logs['train_prf'].append(m[:-1])
                else:
                    self.loss_logs['train_prf'] = [m[:-1]]
                if m[2] > best_train_fscore:
                    best_train_fscore = m[2]
                    self.final_train_fscore = m[2]
                    torch.save(self.devnet.state_dict(), 'checkpoints/' + self.name + '_' + str(self.gpu) + '_' + 'checkpoint.pt')
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print ("early stoppong")
                        self.devnet.load_state_dict(torch.load('checkpoints/' + self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt'))
                        break
                if best_train_fscore > early_stop_fscore:
                    break
        torch.save(self.devnet.state_dict(), self.name+'.pt')

    def load(self, model_file: str = None):
        if model_file is None:
            self.devnet.load_state_dict(torch.load(self.name+'.pt'))
        else:
            self.devnet.load_state_dict(torch.load(model_file))

    def predict(self, datalist: list):
        data_loader = DataLoader(dataset=datalist, batch_size=self.batch_size, shuffle=False)

        self.devnet.eval()

        outputs = []
        features = []

        for (i, batch) in enumerate(tqdm(data_loader)):
            output, feature = self.devnet(batch)

            outputs.append(output.data.cpu().numpy())
            features.append(feature.data.cpu().numpy())

        outputs = np.concatenate(outputs)
        features = np.concatenate(features)

        return outputs, features


class SpatioDevNetModule(nn.Module, PyTorchUtils):
    def __init__(self, input_dim: int, hidden_dim: int = 4, edge_attr_len: int = 60, global_fea_len: int = 2,num_layers: int = 2,
                 edge_module: str = 'linear', act: bool = True, pooling: str = 'attention', is_bilinear: bool = False, nonlinear_scorer: bool = False,
                 head: int = 4, aggr: str = 'mean', concat: bool = False, dropout: float = 0.5, seed: int = 0, gpu: int = None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_len = edge_attr_len
        self.global_fea_len = global_fea_len
        self.num_layers = num_layers

        self.edge_module = edge_module
        self.act = act
        self.pooling = pooling
        self.is_bilinear = is_bilinear
        self.nonlinear_scorer = nonlinear_scorer
        self.head = head
        self.aggr = aggr
        self.concat = concat
        self.dropout = dropout

        if self.edge_module == 'lstm':
            self.intermediate = NNConv(self.input_dim, self.hidden_dim,
                                       LSTMhelper(self.edge_attr_len, self.input_dim * self.hidden_dim), self.aggr)
        else:
            self.intermediate = NNConv(self.input_dim, self.hidden_dim,
                                       nn.Linear(self.edge_attr_len, self.input_dim * self.hidden_dim), self.aggr)
        self.to_device(self.intermediate)

        self.local_scorer = GATConv(self.hidden_dim, self.global_fea_len, self.head, self.concat, dropout=self.dropout)
        self.to_device(self.local_scorer)

        self.attention_pooling = GlobalAttention(nn.Linear(self.global_fea_len, 1)) if self.pooling == 'attention' else None
        if self.attention_pooling is not None:
            self.to_device(self.attention_pooling)

        if self.nonlinear_scorer:
            self.final_scorer_res = nn.Sequential(
                                  nn.Linear(2 * self.global_fea_len, 4 * self.global_fea_len),
                                  nn.LeakyReLU(0.1),
                                  nn.Dropout(p = self.dropout),
                                  nn.Linear(4 * self.global_fea_len, self.global_fea_len)
                                )
            self.final_scorer = nn.Linear(self.global_fea_len, 1)
        else:
            if self.is_bilinear:
                self.final_scorer = nn.Bilinear(self.global_fea_len, self.global_fea_len, 1)
            else:
                self.final_scorer = nn.Linear(2 * self.global_fea_len, 1)
        self.to_device(self.final_scorer)

    def forward(self, data):
        representation = self.intermediate(data.x, data.edge_index, data.edge_attr)

        if self.act:
            representation = F.relu(representation)
        representation = F.dropout(representation, p = self.dropout, training = self.training)

        scores = self.local_scorer(representation, data.edge_index)

        if self.pooling == 'max':
            local_score_summary =  global_max_pool(scores, data.batch)
        elif self.pooling == 'mean':
            local_score_summary =  global_mean_pool(scores, data.batch)
        elif self.pooling == 'add':
            local_score_summary =  global_add_pool(scores, data.batch)
        else:
            local_score_summary =  self.attention_pooling(scores, data.batch)
        if self.nonlinear_scorer:
            res = self.final_scorer_res(torch.cat((local_score_summary, data.global_x), 1))
            final_repr = local_score_summary + res
            return self.final_scorer(final_repr), final_repr
        else:
            if self.is_bilinear:
                return self.final_scorer(local_score_summary, data.global_x), torch.cat((local_score_summary, data.global_x), 1)
            else:
                return self.final_scorer(torch.cat((local_score_summary, data.global_x), 1)), torch.cat((local_score_summary, data.global_x), 1)

    @staticmethod
    def deviation_loss_function(preds, labels, confidence_margin):
        ref = torch.normal(mean=0., std=1.0, size=(5000,))
        dev = (preds - torch.mean(ref)) / torch.std(ref)
        dev = dev.squeeze()
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.max(confidence_margin - dev, torch.zeros_like(confidence_margin)))
        return torch.mean((1 - labels) * inlier_loss + labels * outlier_loss)

    @staticmethod
    def cross_entropy_loss_function(preds, labels):
        sig = nn.Sigmoid()
        loss = nn.BCELoss()
        return loss(sig(preds),labels)

    @staticmethod
    def bce_focal_loss_function(preds, labels, alpha = 0.5, gamma = 0.5):
        pt = torch.sigmoid(preds)
        loss = - 2 * alpha * (1 - pt) ** gamma * labels * torch.log(pt) - \
               2 * (1 - alpha) * pt ** gamma * (1 - labels) * torch.log(1 - pt)
        loss = torch.mean(loss)
        return loss

class LSTMhelper(nn.Module, PyTorchUtils):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 seed: int = 0, gpu: int = None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first=True)

    def forward(self, data):
        return self.lstm(data)[1][0][-1]
