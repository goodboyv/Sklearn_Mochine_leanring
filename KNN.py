"""
    KNN算法也叫做K近邻算法，它的主要思想是：
        计算测试样本与训练集中各个样本之间的距离，选择与测试样本距离最近的K个，然后统计这K个样本中出现标记最多的那个，
        将这个标记作为测试样本的标记
"""

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn():
    # 加载数据集
    iris = load_iris()
    feature = iris.data
    target = iris.target
    print("特征名称：", iris.feature_names)
    print("目标标记名：", iris.target_names)
    print("特征：", feature.shape)
    print("标记：", target.shape)
    # 特征预处理
    # 判断有没有缺失值
    print(pd.isnull(feature).any())
    # 标准化
    std = StandardScaler()
    feature = std.fit_transform(feature)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)
    # 建立KNN模型
    kn = KNeighborsClassifier(n_neighbors=5)
    # 训练
    kn.fit(x_train, y_train)
    # 验证
    score_val = kn.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    # 测试
    score_test = kn.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    predict = kn.predict(x_test)
    print("在测试集上的预测结果：", predict)


if __name__ == "__main__":
    knn()
