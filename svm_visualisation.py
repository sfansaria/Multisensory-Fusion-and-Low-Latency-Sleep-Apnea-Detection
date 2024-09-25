import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def load_dic(path):
    df = pd.read_pickle(path)
    return df


def visualise_style1(x,y,svc):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface') #of linear SVC
    # Set-up grid for plotting.
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()

def visualise_style2(x,y,svc):
    ax = plt.gca()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    ax.set_title("kernal boundary and overlap") #SVC
    plt.show()


if __name__ == '__main__':
    dic1 = "/home/asif/PycharmProjects/sba/save/features_shuffle"
    df1 = load_dic(dic1)
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
    # x = x[:,11:22]

    x = StandardScaler().fit_transform(x)

    print(x.shape)
    x = x[:,11:]
    # exit(-1)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)

    x = pca_result

    print(x.shape)

    # exit(-1)

    length = len(x)
    print(length)
    train_num = int(length*0.85)


    x_train = x[:train_num]
    y_train = y[:train_num]

    x_test = x[train_num:]
    y_test = y[train_num:]
    print(len(x_train),len(x_test),len(x))
    print(len(x_train[0]),len(x_test[0]),len(x[0]))

    # Instantiate the Support Vector Classifier (SVC)
    svc = SVC(C=1.0, random_state=1, kernel='linear')

    # Fit the model
    svc.fit(x_train, y_train)

    # Make the predictions
    y_predict = svc.predict(x_test)

    # Measure the performance
    print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))
    print(metrics.classification_report(y_test, y_predict))
    visualise_style1(x_test, y_test, svc)
    visualise_style2(x_test,y_test,svc)





