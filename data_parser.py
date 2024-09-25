import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def file_parser(directory_path):
    filelist = os.listdir(directory_path)
    for file in filelist:
        filepath = os.path.join(directory_path,file)
        with open(filepath, "r" ) as f1:
            count = 0
            for _ in range(5):
                next(f1)
            for line in f1:
                count = count+1
        print(file, count)


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

def wavelet_coefficient_generation(samples,time,mode, title):
    # normalised_samples = preprocessing.normalize(np.array(samples).reshape(-1,1))
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(np.array(samples).reshape(-1,1))
    normalised_samples = scaler.transform(np.array(samples).reshape(-1,1))
    # print(normalised_samples.shape)
    cA, cD = pywt.dwt(normalised_samples,'db2')
    coefficients = (cA,cD)
    for i, ci in enumerate(coefficients):
        plt.imshow(ci.reshape(1, -1), extent=[0, normalised_samples.shape[0], i + 0.5, i + 1.5], cmap='inferno', aspect='auto',
                   interpolation='nearest')
    plt.plot(normalised_samples, ls='dotted', c='red', lw=1)
    plt.ylim(0.5, len(coefficients) + 0.5)  # set the y-lim to include the six horizontal images
    # optionally relabel the y-axis (the given labeling is 1,2,3,...)
    plt.yticks(range(1, len(coefficients) + 1), ['cA', 'cD'])
    plt.title(title)
    # plt.savefig(path)
    plt.show()

def signal_visualisation(samples, time, title):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(np.array(samples).reshape(-1, 1))
    normalised_samples = scaler.transform(np.array(samples).reshape(-1, 1))
    normalised_samples = normalised_samples[200:-1000]
    plt.plot(normalised_samples, ls='dotted', c='red', lw=1)
    plt.title(title)
    plt.ylabel("Normalised signal")
    plt.xlabel("time signature")
    plt.show()




if __name__ == '__main__':
    directory_path = "/home/asif/Documents/sba/data/ChildData/PI001/Analysed data"
    file_parser(directory_path)
    # filepath = "/home/asif/Documents/sba/data/ChildData/PI001/Raw data/SPO2 - PI001.txt"
    # sample, time = sample_extraction(filepath)
    # for i,j in zip(sample,time):
    #     print(i,j)
    # print((sample))
    # print((sample))
    # signal_visualisation(sample,time,"SpO2 normalised")
    # wavelet_coefficient_generation(sample,time,"db1","SpO2 normalised")

    # filepath = "/home/asif/Documents/sba/data/ChildData/PI001/Raw data/Sum RIPs - PI001.txt"
    # sample, time = sample_extraction(filepath)
    # signal_visualisation(sample, time, "SUM RIPs normalised")
    # wavelet_coefficient_generation(sample, time, "db1", "SUM RIPs normalised")

    filepath = "/home/asif/Documents/sba/data/ChildData/PI001/Raw data/CO2 - PI001.txt"
    sample, time = sample_extraction(filepath)
    # print(len(sample))
    # print(len(sample))
    #
    signal_visualisation(sample, time, "Co2 normalised")
    # wavelet_coefficient_generation(sample, time, "db1", "Co2 normalised")