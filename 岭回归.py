"""
    带有正则化的线性回归
"""


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def ridge():
    # 加载数据集
    boston = load_boston()
    feature = boston.data
    target = boston.target

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)

    # 标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 建立模型
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)

    # 预测结果
    y_predict = rd.predict(x_test)
    y_predict_inverse = std_y.inverse_transform(y_predict)
    print(y_predict_inverse)

    # 均方误差
    error = mean_squared_error(y_test, y_predict)
    print("均分误差：", error)


if __name__ == "__main__":
    ridge()
