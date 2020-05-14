"""
    支持向量机：通过寻找划分超平面来进行分类的算法，这个划分超平面只由支持向量有关，与其他样本无关
"""


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def svm():
    # 加载数据集
    iris = load_iris()

    # 取出特征值和目标值
    feature = iris.data
    target = iris.target
    print("特征：", iris.feature_names)
    print("目标：", iris.target_names)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 建立模型
    clf = SVC(kernel="linear", C=0.4)
    # 训练
    clf.fit(x_train, y_train)
    # 验证
    score_val = clf.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    # 测试
    score_test = clf.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    # 预测
    predict = clf.predict(x_test)
    print("预测结果：", predict)
    # 打印召回率、F1
    print(classification_report(y_test, predict, labels=[0, 1, 2], target_names=iris.target_names))


if __name__ == "__main__":
    svm()


