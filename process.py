import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder #只允许二维以上的数据进行输入
def process():
    data = pd.read_csv('kdd.csv')
    oe = OrdinalEncoder()
    # 利用训练集进行fit  非数值转化数值
    oe = oe.fit(data.iloc[:, 1:4])
    data.iloc[:, 1:4] = oe.transform(data.iloc[:, 1:4])
    kdd_types = data.iloc[:, -1].unique()  # ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
    replace_list=[]
    for item in kdd_types:
        if item == 'normal. ':
            replace_list.append(1)
        else:
            replace_list.append(0)
    # 多分类转化为二分类
    # print(len(kdd_types))
    # print(len(replace_list))
    data = data.replace(kdd_types, replace_list)
    print()
    x_train = data.drop(columns=data.columns[len(data.columns)-1], axis=1)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据转换为均值为0，方差为1的数据
    ss = StandardScaler()
    ss = ss.fit(x_train)
    x = ss.transform(x_train)

    # 降维处理
    pca = PCA(n_components=10)
    pca.fit(x_train)
    plt.figure()
    plt.plot([i for i in range(x_train.shape[1])],
             [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(x_train.shape[1])])
    # plt.show()
    feature = pca.transform(x_train)

    # 标签与特征整合
    label = data[data.columns[len(data.columns)-1]]
    label = DataFrame(label)
    feature = DataFrame(feature)
    data = pd.concat([feature, label], axis=1)

    pd.DataFrame(data).to_csv('newdata.csv')
if __name__ == '__main__':
    process()