import os

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn import preprocessing


def signal_visualisation(samples, time, title):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(np.array(samples).reshape(-1, 1))
    normalised_samples = scaler.transform(np.array(samples).reshape(-1, 1))
    normalised_samples = normalised_samples[:-1000]
    plt.plot(normalised_samples, ls='dotted', c='red', lw=1)
    # plt.plot(samples, ls='dotted', c='red', lw=1)
    plt.title(title)
    plt.ylabel("Normalised signal")
    plt.xlabel("time signature")
    plt.show()


def sample_extraction(filepath):
    from scipy import stats

    samples = []
    time = []
    sample_rate = 1
    with open(filepath,"r") as f:
        for _ in range(2):
            next(f)
        for line in f:
            sample_rate = int(line.strip().split(":")[1].strip())
            break
        for _ in range(4):
            next(f)
        counter = 0
        mean = 0
        sensor = filepath.split("/")[-1]
        print("sample rate in {} is {}".format(sensor,sample_rate))
        time_prev = 0
        for line in f:
            timestamp = line.strip().split(";")[0].split(",")[0]
            if counter == sample_rate:
                mean = float(mean/sample_rate)
                counter = 0
                time.append(time_prev)
                samples.append(mean)
                mean = 0
            # print(line.strip().split(";")[1].strip())
            mean = mean + float(line.strip().split(";")[1].strip())
            counter = counter + 1
            time_prev = timestamp

    print(len(samples),len(time))
    pivot = 0
    for i in range(len(time)):
        if "06:56:59" in time[i]:
            break
        else:
            pivot+=1
    samples = samples[:pivot]
    time = time[:pivot]
    return samples,time


def Isolation_forest(sample,time,sensor,plot=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest

    original_sample = sample
    outliers_fraction = float(.003)
    plt.rc('figure', figsize=(12, 6))
    plt.rc('font', size=15)
    original_sample = np.array(original_sample)
    sample = np.array(sample)
    scaler = StandardScaler()
    # np_scaled = scaler.fit_transform(sample.reshape(-1, 1))
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(sample.reshape(-1, 1))
    anomaly = model.predict(sample.reshape(-1,1))
    # model.fit(np_scaled)
    # anomaly = model.predict(np_scaled)
    # for i,j in zip(sample,time):
    #     print(i,j)
    print(anomaly.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    anomaly_points = []
    anomaly_times = []
    normal_points = []
    nomral_times = []
    for i in range(anomaly.shape[0]):
        if anomaly[i]==-1:
            # print(i,anomaly[i],sample[i],original_sample[i])
            anomaly_points.append(original_sample[i])
            anomaly_times.append(i)
        else:
            normal_points.append(original_sample[i])
            nomral_times.append(i)

    if plot:
        ax.plot(nomral_times, normal_points, color='green', label='regular')
        # ax.plot(anomaly_times, anomaly_points, color='red', label='anomaly')
        # ax.scatter(nomral_times,normal_points,  color='green', label='regular')

        ax.scatter(anomaly_times, anomaly_points, color='red', label='irregular')
        plt.legend(loc="lower left")
        title = sensor.split("/")[-1]
        plt.title(title)
        plt.show()

    return anomaly


def multi_sensor_visualisation(rootdir,sensornames):
    filelist = os.listdir(rootdir)
    for file in filelist:
        counter = 0
        for i in sensornames:
            if i.split(" ")[0] in file.split(" ")[0]:
                filepath = os.path.join(rootdir,file)
                print(i)
                sample, time = sample_extraction(filepath)
                print(i,len(sample),len(time))
                Isolation_forest(sample,time,i)
                counter+=1


def filter_noise(sample, time):
    new_sample = []
    new_time = []
    removed_time = []
    for i in range(len(sample)):
        if sample[i] <= 100 and sample[i] > 0:
            new_sample.append(sample[i])
            new_time.append(time[i])
        else:
            removed_time.append(time[i])
    return new_sample,new_time, removed_time


def remove_indexes(sample, time, removed_time):
    new_sample = []
    new_time = []
    for i in range(len(time)):
        if time[i] not in removed_time:
            new_sample.append(sample[i])
            new_time.append(time[i])
    return new_sample,new_time


def multi_sensor_fusion(rootdir, primary_sensorname, auxilary_sensorname, save=False):
    import itertools
    import random
    import pickle

    filelist = os.listdir(rootdir)
    for file in filelist:
        if primary_sensorname in file:
            primary_sensor = os.path.join(rootdir,file)
            break
    primary_sensor_sample,primary_sensor_time = sample_extraction(primary_sensor)
    # signal_visualisation(primary_sensor_sample, primary_sensor_time, "SPO2 Normalised before noise removal")
    # exit(-1)
    primary_sensor_sample, primary_sensor_time, removed_time = filter_noise(primary_sensor_sample,primary_sensor_time)
    # print(removed_time)
    # signal_visualisation(primary_sensor_sample, primary_sensor_time, "SPO2 Normalised after noise removal")
    # exit(-1)
    primary_anomaly = Isolation_forest(primary_sensor_sample,primary_sensor_time,
                                       primary_sensor,plot=True)
    # exit(-1)
    fused_features_anomaly = {}
    fused_features_normal = {}
    primary_feature_samples_num = len(primary_anomaly)
    context_size = 5

    anomaly_time = []
    normal_time = []
    for i in range(len(primary_anomaly)):
        if primary_anomaly[i] == -1:
            if (i-context_size-1 > 0) and (i+context_size+1 < primary_feature_samples_num):
                back_context = primary_sensor_sample[i-context_size:i]
                front_context = primary_sensor_sample[i+1:i+context_size+1]
                # print(len(back_context),len(front_context))
                # print(primary_sensor_sample[i-5:i+5])
                # print(back_context,primary_sensor_sample[i],front_context)
                # print(primary_sensor_time[i])
                feature = list(itertools.chain(back_context,[primary_sensor_sample[i]],front_context))
                fused_features_anomaly[primary_sensor_time[i]] = feature
                anomaly_time.append(primary_sensor_time[i])

    # print(anomaly_time)
    # print(fused_features_anomaly)
    anomaly_features_number = len(fused_features_anomaly)
    normal_samples_number = anomaly_features_number + 5
    normal_samples_index = random.sample(range(10,len(primary_anomaly)-context_size-1),normal_samples_number+30)

    counter = 0
    for i in normal_samples_index:
        if primary_anomaly[i]==1:
            back_context = primary_sensor_sample[i - context_size:i]
            front_context = primary_sensor_sample[i + 1:i + context_size + 1]
            # print(len(back_context),len(front_context))
            # print(primary_sensor_sample[i-5:i+5])
            # print(back_context,primary_sensor_sample[i],front_context)
            # print(primary_sensor_time[i])
            feature = list(itertools.chain(back_context, [primary_sensor_sample[i]], front_context))
            fused_features_normal[primary_sensor_time[i]] = feature
            normal_time.append(primary_sensor_time[i])
            counter+=1
            if counter > normal_samples_number:
                break

    # print(fused_features_normal)

    filelist = os.listdir(rootdir)
    for file in filelist:
        counter = 0
        for i in auxilary_sensorname:
            if i.split(" ")[0] in file.split(" ")[0]:
                print(file)
                filepath = os.path.join(rootdir,file)
                # print(i)
                sample, time = sample_extraction(filepath)
                sample, time = remove_indexes(sample, time, removed_time)
                print(i,len(sample),len(time))
                # Isolation_forest(sample,time,i)
                for i in range(len(primary_anomaly)):
                    if primary_anomaly[i] == -1:
                        if (i - context_size - 1 > 0) and (i + context_size + 1 < primary_feature_samples_num):
                            back_context = sample[i - context_size:i]
                            front_context = sample[i + 1:i + context_size + 1]
                            # print(len(back_context),len(front_context))
                            # print(sample[i-5:i+5])
                            # print(back_context,sample[i],front_context)
                            # print(time[i])
                            feature = list(itertools.chain(back_context, [sample[i]], front_context))
                            if time[i] in fused_features_anomaly:
                                prev_features = fused_features_anomaly[time[i]]
                                new_feature_list = list(itertools.chain(prev_features, feature))
                                fused_features_anomaly[time[i]] = new_feature_list
                            else:
                                print(time[i]," is not present")
                                exit(-1)

                # print(fused_features_anomaly)
                counter = 0
                for i in normal_samples_index:
                    if primary_anomaly[i] == 1:
                        back_context = sample[i - context_size:i]
                        front_context = sample[i + 1:i + context_size + 1]
                        # print(len(back_context),len(front_context))
                        # print(primary_sensor_sample[i-5:i+5])
                        # print(back_context,primary_sensor_sample[i],front_context)
                        # print(primary_sensor_time[i])
                        feature = list(itertools.chain(back_context, [sample[i]], front_context))
                        if time[i] in fused_features_normal:
                            prev_features = fused_features_normal[time[i]]
                            new_feature_list = list(itertools.chain(prev_features, feature))
                            fused_features_normal[time[i]] = new_feature_list
                        else:
                            print(time[i], " is not present")
                            exit(-1)
                        counter += 1
                        if counter > normal_samples_number:
                            break
                # print(fused_features_normal)
                counter+=1
    print(len(fused_features_anomaly))
    print(len(fused_features_normal))
    anomaly_path = "save/"+rootdir.split("/")[-3]+"_fused_features_anomaly.pkl"
    normal_path = "save/"+rootdir.split("/")[-3]+"_fused_features_normal.pkl"
    if save:
        with open(anomaly_path, 'wb') as f:
            pickle.dump(fused_features_anomaly, f)
        with open(normal_path, 'wb') as f:
            pickle.dump(fused_features_normal, f)



if __name__ == '__main__':
    # filepath = "/home/asif/Documents/sba/data/ChildData/PI001/Raw data/SPO2 - PI001.txt"
    # sample, time = sample_extraction(filepath)
    # # print(sample)
    # Isolation_forest(sample,time)
    rootdir = "/home/asif/Documents/sba/data/ChildData/PI006/Raw data/"
    auxilary_sensorname = ["CO2","Pulse","Sum RIPS","Snore"]
    primary_sensorname = "SPO2"
    multi_sensor_fusion(rootdir, primary_sensorname, auxilary_sensorname, save=False)