"""
    逻辑回归：将线性回归函数的输出，作为Sigmoid函数的输入，然后输出为0-1之间的
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def LR():
    # 加载数据集
    cancer = pd.read_csv("E:/Desktop/机器学习_新/数据集/癌症数据集/Prostate_Cancer.csv")
    pd.set_option("display.max_columns", 100)
    print(cancer.head(5))
    print("特征值名称：", list(cancer.columns))

    # 提取特征值和目标值
    feature = cancer[list(cancer.columns)[2:]]
    print(feature.head(5))
    target = cancer[list(cancer.columns)[1]]
    print(target.head(5))

    # 将目标值进行0-1化
    target.replace("M", 0, inplace=True)
    target.replace("B", 1, inplace=True)
    print(target.head(5))

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_val = std.transform(x_val)
    x_test = std.transform(x_test)

    # 建立模型
    lg = LogisticRegression()
    # 训练
    lg.fit(x_train, y_train)
    # 验证
    score_val = lg.score(x_val, y_val)
    print("在验证集上的得分：", score_val)
    # 测试
    score_test = lg.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    # 预测
    predict = lg.predict(x_test)
    print(predict)
    # 打印召回率，F1
    print(classification_report(y_test, predict, labels=[0, 1], target_names=["良性", "恶性"]))


if __name__ == "__main__":
    LR()
