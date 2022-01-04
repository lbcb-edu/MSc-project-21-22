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
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate for SGD (default: 0.01)", metavar="")
    parser.add_argument("--momentum", default=0.9, type=float, help="learning rate for SGD (default: 0.9)", metavar="")
    parser.add_argument("--cuda", action="store_true", help="use CUDA for model training.")
    
    return parser.parse_args()

class OverlapModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_inputs, 10),
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

def load_data(path):
    inputs, labels = [], []
    p, n = 0,0
    with open(path, "r") as file:
        for line in file:
            s_l = line.split(" ")
            d = list(map(lambda x: float(x), s_l[:3]))
            l = float(s_l[3])
            # if l == 0:
            #     n += 1
            # else:
            #     p += 1
            # if n >= 35903 and l == 0:
            #     continue
            inputs.append(d)
            labels.append([l])
    # print(sum(1 for item in labels if item == [1]))
    # print(sum(1 for item in labels if item == [0]))
    inputs, labels = torch.tensor(inputs), torch.tensor(labels)
    c_min = inputs.min(0, keepdim=True)[0]
    inputs -= c_min
    inputs /= inputs.max(0, keepdim=True)[0] - c_min
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.4, shuffle=True)
    train_dl = DataLoader(OverlapLoader(x_train, y_train), batch_size=32, shuffle=True)
    test_dl = DataLoader(OverlapLoader(x_test, y_test), batch_size=1024, shuffle=False)
    return train_dl, test_dl

def train(model, train_data_loader, test_data_loader, learning_rate, momentum, device):
    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        if (epoch % 10 == 0):
            acc = evaluate(model, test_data_loader, device)
            print("Epoch = " + str(epoch) + ", acc = " + str(acc))


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

if __name__=="__main__":

    args = parse_arguments()

    if (args.cuda and not torch.cuda.is_available()):
        raise ValueError("CUDA is not avaialble")
    else:
        args.device = torch.device("cuda" if args.cuda else "cpu")

    train_dl, test_dl = load_data("dataset.txt")
    model = OverlapModel(3)
    model.to(args.device)
    train(model, train_dl, test_dl, args.learning_rate, args.momentum, args.device)
    acc = evaluate(model, test_dl, args.device)
    print(acc)
