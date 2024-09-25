# first neural network with keras tutorial
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


# load the dataset

def load_dic(path):
    df = pd.read_pickle(path)
    return df

def dataprep(datapath):
    df1 = load_dic(datapath)
    x_ = df1.loc[:, ['feature']].values
    y_ = df1.loc[:, ['label']].values

    y = []
    for i in range(y_.shape[0]):
        if y_[i] == "normal":
            y.append(1)
        elif y_[i] == "anomaly":
            y.append(0)
        else:
            exit(-1)

    x = []
    for i in range(x_.shape[0]):
        x.append(x_[i][0])

    # exit(-1)
    x = StandardScaler().fit_transform(x)
    # x = x[:,:11]
    # split into input (X) and output (y) variables

    length = len(x)
    print(length)
    train_num = int(length * 0.85)

    x_train = x[:train_num]
    y_train = y[:train_num]

    x_test = x[train_num:]
    y_test = y[train_num:]

    print(len(x_train), len(x_test), len(x))
    print(len(x_train[0]), len(x_test[0]), len(x[0]))

    return x_train,y_train,x_test,y_test

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class NeuralNetwork_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork_simple, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))

        return x

class CNN_simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.conv3 = nn.Conv1d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 5 , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class CNN_simple2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 8)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, 5)
        self.conv3 = nn.Conv1d(128, 64, 5)
        self.fc1 = nn.Linear(64 * 3 , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CNN_simple_11dim(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.pool = nn.MaxPool1d(2, 1)
        self.conv2 = nn.Conv1d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 3 , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CNN_simple3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.conv2 = nn.Conv1d(1, 64, 5)
        self.conv3 = nn.Conv1d(1, 64, 5)
        self.conv4 = nn.Conv1d(1, 64, 5)
        self.conv5 = nn.Conv1d(1, 64, 5)
        self.x1 = nn.Parameter(torch.rand(1))
        self.x2 = nn.Parameter(torch.rand(1))
        self.x3 = nn.Parameter(torch.rand(1))
        self.x4 = nn.Parameter(torch.rand(1))
        self.x5 = nn.Parameter(torch.rand(1))

        self.convx1 = nn.Conv1d(64, 128, 5)
        self.convx2 = nn.Conv1d(128, 64, 5)
        self.convx3 = nn.Conv1d(64, 64, 5)
        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(64 * 1 , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        x1 = x[:,:,:11]
        x2 = x[:, :, 11:22]
        x3 = x[:, :, 22:33]
        x4 = x[:, :, 33:44]
        x5 = x[:, :, 44:55]
        # print(x.shape)
        # x1 = torch.mul(F.relu(self.conv1(x1)),self.x1)
        x1 = torch.mul(F.relu(self.conv1(x1)), self.x1)
        x2 = torch.mul(F.relu(self.conv2(x2)),self.x2)
        x3 = torch.mul(F.relu(self.conv3(x3)),self.x3)
        x4 = torch.mul(F.relu(self.conv4(x4)),self.x4)
        x5 = torch.mul(F.relu(self.conv5(x5)),self.x5)


        x = torch.cat((x1,x2,x3,x4,x5),dim=2)
        print(self.x1,self.x2,self.x3,self.x4,self.x5)
        # print(x.shape)
        x = self.pool(F.relu(self.convx1(x)))
        x = self.pool(F.relu(self.convx2(x)))
        x = F.relu(self.convx3(x))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class CNN_simple_sensors(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3)
        self.conv2 = nn.Conv1d(1, 64, 3)
        self.conv3 = nn.Conv1d(1, 64, 3)
        self.conv4 = nn.Conv1d(1, 64, 3)
        self.conv5 = nn.Conv1d(1, 64, 3)
        self.x1 = nn.Parameter(torch.rand(1))
        self.x2 = nn.Parameter(torch.rand(1))
        self.x3 = nn.Parameter(torch.rand(1))
        self.x4 = nn.Parameter(torch.rand(1))
        self.x5 = nn.Parameter(torch.rand(1))

        self.convx1 = nn.Conv1d(64, 128, 3)
        self.convx2 = nn.Conv1d(128, 64, 3)
        self.convx3 = nn.Conv1d(64, 64, 3)
        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(64 * 5 , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # x1 = x[:,:,:11]
        x2 = x[:, :, 11:22]
        x3 = x[:, :, 22:33]
        x4 = x[:, :, 33:44]
        x5 = x[:, :, 44:55]
        # print(x.shape)
        # x1 = torch.mul(F.relu(self.conv1(x1)),self.x1)
        # x1 = torch.mul(F.relu(self.conv1(x1)), self.x1)
        x2 = torch.mul(F.relu(self.conv2(x2)),self.x2)
        x3 = torch.mul(F.relu(self.conv3(x3)),self.x3)
        x4 = torch.mul(F.relu(self.conv4(x4)),self.x4)
        x5 = torch.mul(F.relu(self.conv5(x5)),self.x5)


        x = torch.cat((x2,x3,x4,x5),dim=2)
        print(self.x2,self.x3,self.x4,self.x5)
        # print(x.shape)
        x = self.pool(F.relu(self.convx1(x)))
        x = self.pool(F.relu(self.convx2(x)))
        x = F.relu(self.convx3(x))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class NeuralNetwork_four_sensor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork_four_sensor, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x1 = x[:,:,:11]
        print(x.shape)
        x = x[:,44:55]
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))

        return x


if __name__ == '__main__':
    dic1 = "/home/asif/PycharmProjects/sba/save/features_shuffle"
    X_train,y_train,X_test,y_test = dataprep(dic1)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    batch_size = 100
    # Instantiate training and test data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    input_dim = 11
    hidden_dim = 128
    output_dim = 1

    # model = NeuralNetwork_simple(input_dim, hidden_dim, output_dim)
    # model = NeuralNetwork_four_sensor(input_dim, hidden_dim, output_dim)

    # print(model)
    model = CNN_simple3()
    # model = CNN_simple_sensors()
    print(model)

    learning_rate = 0.001

    loss_fn = nn.BCELoss()
    print("params:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 100
    loss_values = []

    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            # print(pred.shape)
            # print(y.unsqueeze(-1))
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

    print("Training Complete")

    """
    Training Complete
    """

    # step = np.linspace(0, 100, 10500)
    #
    # fig, ax = plt.subplots(figsize=(8, 5))
    # plt.plot(step, np.array(loss_values))
    # plt.title("Step-wise Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

    """
    We're not training so we don't need to calculate the gradients for our outputs
    """
    y_pred = []
    y_hyp = []
    total = 0
    correct = 0
    time_start = time.time()

    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            y_hyp.append(y)
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()

    print('NN done! Time elapsed: {} seconds'.format(time.time() - time_start))
    print("Accuracy of the network on the {} test instances is:{}".format(total,(100 * correct // total)))

    y_pred = list(itertools.chain(*y_pred))
    y_hyp = list(itertools.chain(*y_hyp))

    print(classification_report(y_hyp, y_pred))

    cf_matrix = confusion_matrix(y_hyp, y_pred)

    plt.subplots(figsize=(8, 5))

    sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

    plt.show()