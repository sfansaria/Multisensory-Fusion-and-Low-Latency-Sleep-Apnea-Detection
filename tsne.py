from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt



def load_dic(path):
    df = pd.read_pickle(path)
    return df


if __name__ == '__main__':
    dic1 = "/home/asif/PycharmProjects/sba/save/features_shuffle"
    df1 = load_dic(dic1)

    x_ = df1.loc[:, ['feature']].values
    y_ = df1.loc[:, ['label']].values

    y = []
    for i in range(y_.shape[0]):
        if y_[i] =="normal":
            y.append(1)
        elif y_[i] == "anomaly":
            y.append(0)
        else:
            exit(-1)
    # exit(-1)
    # print(y)

    # exit(-1)

    x = []
    for i in range(x_.shape[0]):
        x.append(x_[i][0])

    # exit(-1)
    x = StandardScaler().fit_transform(x)
    # x = x[:,11:]

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(x)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300) #40
    tsne_results = tsne.fit_transform(x)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # exit(-1)
    df_subset = {}

    df_subset['pca-one'] = pca_result[:, 0]
    df_subset['pca-two'] = pca_result[:, 1]
    df_subset['pca-three'] = pca_result[:, 2]

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]

    df_subset['y'] = y
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=0.9
    )

    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax2
    )

    plt.show()






