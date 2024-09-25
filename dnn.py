# first neural network with keras tutorial
import time

import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def build_model(model_name):
    if "simple_dnn" in model_name:
        model = Sequential()
        model.add(Dense(128, input_shape=(55,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    elif "feedforward_deep" in model_name:
        model = Sequential()
        model.add(Dense(128, input_shape=(55,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    elif "lstm_feedforward" in model_name:
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(55, 1)))
        # model.add(Dropout(0.2))

        # model.add(LSTM(units=128, return_sequences=True))
        # model.add(Dropout(0.25))

        # model.add(LSTM(units=256, return_sequences=True))
        # model.add(Dropout(0.25))

        # model.add(LSTM(units=50))
        # model.add(Dropout(0.25))

        model.add(Dense(units=1))
        return model


def dnn_train_eval(datapath, model="simple_dnn"):

    x_train, y_train, x_test, y_test = dataprep(datapath)

    X = np.asarray(x_train)
    y = np.asarray(y_train)

    print(X.shape, y.shape)
    # define the keras model
    model = build_model(model_name=model)
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.count_params())
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=200)
    # evaluate the keras model
    time_start = time.time()
    _, accuracy = model.evaluate(np.asarray(x_test), np.asarray(y_test))
    print('NN done! Time elapsed: {} seconds'.format(time.time() - time_start))
    print('Accuracy: %.2f' % (accuracy * 100))


if __name__ == '__main__':
    dic1 = "/home/asif/PycharmProjects/sba/save/features_shuffle"
    modelnames = ["simple_dnn", "feedforward_deep", "lstm_feedforward"]
    dnn_train_eval(dic1, model="simple_dnn")
