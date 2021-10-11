# coding=utf8
import torch
import sys
import os
import pandas as pd
import numpy as np

import torchvision
from torch import optim
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn import preprocessing


class Arguments():
    """
    set essential arguements for training (Global variables)
    """

    def __init__(self):
        self.batch_size = 64
        self.iteration = 1000
        self.epochs = 10
        self.lr = 0.05
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 10
        self.log_interval = 100
        self.save_model = False

        self.root_path = "C:/Users/Laplace/OneDrive - University of Exeter/Dataset/cicids2017/MachineLearningCSV/MachineLearningCVE/"
        self.DDoS = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        self.Bot = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        self.PortScan = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"


args = Arguments()

src_name = args.DDoS  # set src
tgt_name = args.Bot  # set tgt

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # num_worker <= CPU cores

"""
load data and data pre-process
read intrusion detection datasets via pandas
root_path locked in class Arguements()
datasets used here: cicids2017
"""


def data_preprocessor(df):
    df = pd.read_csv(args.root_path + args.DDoS)
    df = df.dropna()
    df_data = pd.DataFrame()
    df_label = pd.DataFrame()
    df_data = df.iloc[:, 0:78].values
    df_label = df.iloc[:, -1].values
    le = preprocessing.LabelEncoder()
    df_label = le.fit_transform(df_label)
    data = torch.from_numpy(np.array(list(df_data), dtype=np.float))
    label = torch.from_numpy(np.array(list(df_label), dtype=np.float))
    return data, label


def make_data(x, y):
    deal_dataset = TensorDataset(x, y);
    loader = DataLoader(dataset=deal_dataset, batch_size=args.batch_size, shuffle=True)
    return loader


def load_data(root_path, file_name):
    Raw = pd.read_csv(root_path + file_name)
    data, label = data_preprocessor(Raw)
    data_loader = make_data(data, label)

    return data_loader


class BPNet(nn.Module):
    """
    generate neural network modules for intrusion detection
    - simple nn with 3 hidden layers + 1 out layer (aiming to extend mmd layer for further improve)
    """
    def __init__(self):
        super(BPNet, self).__init__()
        self.hidden1 = nn.Linear(78, 10)
        self.out = nn.Linear(10, 3)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = self.out(x)
        return x


print(BPNet())

src_data = load_data(args.root_path, src_name)
tgt_data = load_data(args.root_path, tgt_name)





def train(args, model, device, src_data, optimizer, loss_fn):
    """
    start model training
    """
    model.train()
    for i in range(1, args.iteration + 1):
        optimizer.zero_grad()
        for data, label in src_data:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = loss_fn(output, label.long())
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, 100. * i / args.iteration, loss.item()))


def test(args, model, device, tgt_data):
    """
    start model testing
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in tgt_data:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(tgt_data.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(tgt_data.dataset),
        100. * correct / len(tgt_data.dataset)))


for i in range(1, args.epochs + 1):
    print("Running in epoch {}".format(i))
    model = BPNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    # SGD: random gradient decend
    loss_fn = nn.CrossEntropyLoss()
    model = model.float()
    train(args, model, device, src_data, optimizer, loss_fn)
    test(args, model, device, src_data)