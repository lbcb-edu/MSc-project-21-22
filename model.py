import re
import os
import argparse
import torch
import torch.nn as nn
from numpy import vstack
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from OverlapLoader import OverlapLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="train dataset file")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate for SGD (default: 0.01)", metavar="")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD (default: 0.9)", metavar="")
    parser.add_argument("--cuda", action="store_true", help="use CUDA for model training")
    parser.add_argument('--overlaps', type=argparse.FileType('r'))
    parser.add_argument("--save", type=str, help="save trained model")
    parser.add_argument("--load", type=str, help="load trained model")
    
    return parser.parse_args()

class OverlapModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_inputs),
            nn.Linear(n_inputs, 15),
            nn.ReLU(inplace=True),
            nn.Linear(15, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.classifier.apply(weights_init)

    def forward(self, X):
        probs = self.classifier(X)
        return probs

def parse_cigar(cigar):
    M, I, D = 0,0,0
    b = ""
    for i in cigar:
        if i.isdigit():
            b += i
        else:
            x = int(b)
            if i == "M":
                M += x
            elif i == "I":
                I += x
            elif i == "D":
                D += x
            b = ""
    return M, I, D

def load_data(path):
    inputs, labels = [], []
    with open(path, "r") as file:
        for line in file:
            s_l = line.split(" ")
            d = list(map(lambda x: float(x), s_l[:4]))
            l = float(s_l[5])
            M, I, D = parse_cigar(s_l[4])
            d += [M,I,D]
            inputs.append(d)
            labels.append([l])
    inputs, labels = torch.tensor(inputs), torch.tensor(labels)
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.4, shuffle=True)
    train_dl = DataLoader(OverlapLoader(x_train, y_train), batch_size=32, shuffle=True)
    test_dl = DataLoader(OverlapLoader(x_test, y_test), batch_size=1024, shuffle=False)
    return train_dl, test_dl

def train(model, train_data_loader, test_data_loader, learning_rate, momentum, device):
    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, 201):
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        if (epoch % 10 == 0):
            model.eval()
            with torch.no_grad():
                acc = evaluate(model, test_data_loader, device)
                print("Epoch = " + str(epoch) + ", acc = " + str(acc))
            model.train()


def evaluate(model, data_loader, device):
    inputs, labels = data_loader.dataset[:]
    inputs = inputs.to(device)
    labels = labels.numpy()
    labels = labels.reshape((len(labels), 1))
    predictions = model(inputs).cpu().detach().numpy()
    predictions = predictions.round()
    predictions, labels = vstack(predictions), vstack(labels)
    acc = accuracy_score(labels, predictions)
    return acc

def predict(model, inputs, device):
    inputs = torch.tensor(inputs)
    ids = inputs[:, :1]
    inputs = inputs[:, 1:]
    inputs = inputs.to(device)
    predictions = model(inputs)
    for i in range(len(ids)):
        print(int(ids[i].item()), predictions[i].item())

if __name__=="__main__":

    args = parse_arguments()

    if (args.cuda and not torch.cuda.is_available()):
        raise ValueError("CUDA is not avaialble")
    else:
        args.device = torch.device("cuda" if args.cuda else "cpu")

    if (args.overlaps):
        if (not args.load or not os.path.isfile(args.load)):
            raise ValueError("Model path not valid")
        inputs = []
        model = torch.load(args.load)
        model.eval()
        with args.overlaps as f:
            for line in f:
                src, dst, flag, overlap = line.strip().split(',')
                flag = int(flag)
                if (flag):
                    pattern = r':(\d+)'
                    src_len = int(re.findall(pattern, src.split()[2])[0])
                    overlap = overlap.split()
                    (edge_id, prefix_len), (weight, similarity) = map(int, overlap[:2]), map(float, overlap[2:4])
                    matches = int(overlap[4])
                    cigar = overlap[5]
                    M, I, D = parse_cigar(cigar)
                    inputs.append([edge_id, prefix_len, src_len - prefix_len, similarity, matches, M, I, D])
        predict(model, inputs, args.device)
    else:
        train_dl, test_dl = load_data(args.dataset)
        model = OverlapModel(7)
        model.to(args.device)
        train(model, train_dl, test_dl, args.learning_rate, args.momentum, args.device)
        acc = evaluate(model, test_dl, args.device)
        print(acc)
        if (args.save):
            if not os.path.exists("models"):
                os.mkdir("models")
            torch.save(model, "models/" + args.save)
