import os
import pickle
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

def logits_acc(logits, labels):
    preds = logits.argmax(dim=-1)
    acc = (preds==labels).sum().item()/labels.size(0)
    return acc

class DrebinLoader:
    def __init__(self, root, batch_size, ratio, train=True, train_rate=0.7):
        self.root = root
        self.batch_size = batch_size
        self.malware_num = round(self.batch_size*ratio)
        self.benign_num = self.batch_size - self.malware_num
        self.ignore_types = ["url"]
        self.train = train
        self.train_rate = train_rate
    
    def __iter__(self):
        with open(os.path.join(self.root, "features.pkl"), "rb") as f:
            self.features = pickle.load(f)
        self.num_features = len(self.features)
        apps = set(os.listdir(os.path.join(self.root, "feature_vectors")))
        malwares = np.loadtxt("./drebin/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)
        self.malwares = set(malwares[:, 0].tolist())
        self.benigns = apps.difference(self.malwares)
        del apps, malwares
        if self.train:
            self.malwares = list(self.malwares)[:int(len(self.malwares)*self.train_rate)]
            self.benigns = list(self.benigns)[:int(len(self.benigns)*self.train_rate)]
        else:
            self.malwares = list(self.malwares)[int(len(self.malwares)*self.train_rate):]
            self.benigns = list(self.benigns)[int(len(self.benigns)*self.train_rate):]
        self.malware_length = len(self.malwares)
        self.benign_length = len(self.benigns)
        self.malware_index = 0
        self.benign_index = 0

        return self

    def __next__(self):
        data = torch.zeros((self.batch_size, self.num_features), dtype=torch.float32)
        target = torch.zeros((self.batch_size, ), dtype=torch.long)

        for i in range(self.benign_num):
            self.benign_index += 1
            filename = self.benigns[self.benign_index]
            data[i] = self.vectorize(filename)
            target[i] = 0
        for i in range(self.benign_num, self.benign_num+self.malware_num):
            self.malware_index = (self.malware_index+1)%self.malware_length
            filename = self.malwares[self.malware_index]
            data[i] = self.vectorize(filename)
            target[i] = 1
        if self.benign_index + self.batch_size > self.benign_length:
            raise StopIteration
        else:
            return data, target
    
    def vectorize(self, filename):
        feature_vector = torch.zeros(self.num_features, dtype=torch.float32)
        with open(os.path.join(self.root, "feature_vectors", filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                feature_type = line.split("::")[0]
                if line=="":
                    continue
                if feature_type not in self.ignore_types:
                    feature_vector[self.features[line]] = 1
        return feature_vector

@torch.enable_grad()
def FGSM(model, data, target, epsilon):
    model.eval()
    data_adv = data.clone()
    index = torch.arange(target.size(0)).view(-1, 1)

    data_adv.requires_grad_(True)
    loss = F.cross_entropy(model(data_adv), target)
    grad = torch.autograd.grad(loss, [data_adv])[0]
    data_adv.requires_grad_(False)

    delta = grad.detach()

    values, indices = torch.topk(torch.abs(delta), k=epsilon, dim=-1, largest=True, sorted=False)
    data_adv[index, indices] += torch.sign(values)
    data_adv = torch.clamp(data_adv, min=0.0, max=1.0)

    return data_adv

@torch.enable_grad()
def PGD(model, data, target, steps, epsilon, random_start=False):
    model.eval()
    data_ori = data.detach()
    data_adv = data_ori.clone()
    index = torch.arange(target.size(0)).view(-1, 1)
    for _ in range(steps):
        data_adv.requires_grad_(True)
        loss = F.cross_entropy(model(data_adv), target)
        grad = torch.autograd.grad(loss, [data_adv])[0]
        data_adv.requires_grad_(False)

        delta = data_adv.detach() - data_ori + grad.detach()

        values, indices = torch.topk(torch.abs(delta), k=epsilon, dim=-1, largest=True, sorted=False)
        data_adv[index, indices] += torch.sign(values)
        data_adv = torch.clamp(data_adv, min=0.0, max=1.0)

    return data_adv



class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, dataloader, criterion, optimizer, epoch, eps=None):
    model.train()
    Acc, Loss = 0, 0
    with tqdm(enumerate(dataloader), total=1917, desc=f"Train {epoch+1} eps={eps}") as t:
        for i, (x, y) in t:
            x, y = x.to(device), y.to(device)
            if isinstance(eps, int) and eps>0:
                x = FGSM(net, x, y, eps)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()
            Acc += logits_acc(logits, y)
            t.set_postfix(loss=f"{Loss/(i+1):0.4f}", acc=f"{Acc/(i+1):6.2%}")
    return Loss/(i+1), Acc/(i+1)

@torch.no_grad()
def test(model, dataloader, eps=None):
    model.eval()
    Acc = 0
    with tqdm(enumerate(test_loader), total=606, desc=f"Test eps={eps}") as t:
        for i, (x, y) in t:
            x, y = x.to(device), y.to(device)
            if isinstance(eps, int) and eps>0:
                x = FGSM(net, x, y, eps)
            logits = F.softmax(model(x), dim=-1)
            Acc += logits_acc(logits, y)
            t.set_postfix(acc=f"{Acc/(i+1):6.2%}")
    return Acc/(i+1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drebin")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=6, help='Num of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--eps', type=int, default=None, help='epsilon')
    
    args = parser.parse_args()

    train_loader = DrebinLoader("./drebin", args.batch_size, 0.3, True)
    train_loader = iter(train_loader)
    
    test_loader = DrebinLoader("./drebin", args.batch_size, 0.045, False)
    test_loader = iter(test_loader)

    num_features = train_loader.num_features

    setup_seed(args.seed)
    net = Net(num_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, last_epoch=-1)

    for epoch in range(args.epochs):
        loss, acc = train(net, train_loader, criterion, optimizer, epoch, eps=args.eps)
        scheduler.step()
    acc = test(net, test_loader, eps=None)
    acc = test(net, test_loader, eps=args.eps)
    torch.save(net.cpu().state_dict(), f"eps_{args.eps}.pth")