import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


def load_dic(path):
    df = pd.read_pickle(path)
    return df


if __name__ == '__main__':
    dic1 = "/home/asif/PycharmProjects/sba/save/features_shuffle"
    df1 = load_dic(dic1)

    x_ = df1.loc[:, ['feature']].values
    y = df1.loc[:, ['label']].values

    x = []
    for i in range(x_.shape[0]):
        x.append(x_[i][0])

    # exit(-1)
    x = StandardScaler().fit_transform(x)

    # print(x.shape)
    #
    # df1.to_csv(dic1+"save.csv", sep='\t', encoding='utf-8')
    # exit(-1)
    x = x[:,:11]
    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(x)

    principalDataframe = pd.DataFrame(data=principal_components, columns=['PC1','PC2'])
    finalDf = pd.concat([principalDataframe, df1[['label']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA Cluster', fontsize=20)
    targets = ['anomaly', 'normal']
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                   , finalDf.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    fig.show()






