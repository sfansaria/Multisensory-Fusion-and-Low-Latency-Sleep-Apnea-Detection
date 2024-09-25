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
    x = x[:,:11]
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

    time_start = time.time()
    # Make the predictions
    y_predict = svc.predict(x_test)
    print('SVM done! Time elapsed: {} seconds'.format(time.time() - time_start))
    # Measure the performance
    print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))
    print(metrics.classification_report(y_test, y_predict))

    print("Logistic Regression")

    # Initialize and fit the Model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    time_start = time.time()
    # Make prediction on the test set
    pred = model.predict(x_test)

    print('Logistic regression done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # calculating precision and reall
    precision = metrics.precision_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred)
    print("Accuracy score %.3f" % metrics.accuracy_score(y_test, pred))
    print('Precision: ', precision)
    print('Recall: ', recall)
    print(metrics.classification_report(y_test, pred))
    # Plotting Precision-Recall Curve
    # disp = metrics.plot_precision_recall_curve(model, x_test, y_test)
    # disp.plot()
    # plt.show()



