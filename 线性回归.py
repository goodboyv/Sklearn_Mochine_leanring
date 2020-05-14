"""
    线性回归：通过构建线性模型来进行预测的一种回归算法
"""


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


def linear():
    # 加载数据集
    boston = load_boston()
    feature = boston.data
    target = boston.target

    # 划分数据集
    x_train, x_test, y_train, y_test_ori = train_test_split(feature, target, test_size=0.25)

    # 标准化
    std1 = StandardScaler()
    x_train = std1.fit_transform(x_train)
    x_test = std1.transform(x_test)

    std2 = StandardScaler()
    y_train = std2.fit_transform(y_train.reshape(-1, 1))  # 必须传二维
    y_test = std2.transform(y_test_ori.reshape(-1, 1))

    # 正规方程的解法
    # 建立模型
    lr = LinearRegression()  # 通过公式求解
    lr.fit(x_train, y_train)

    # 预测结果
    y_predict = lr.predict(x_test)  # 这个结果是标准化之后的结果，需要转换
    y_predict_inverse = std2.inverse_transform(y_predict)
    print(y_predict_inverse)

    # 均方误差
    error = mean_squared_error(y_test_ori, y_predict_inverse)
    print("均方误差：", error)

    # 梯度下降算法求解
    sgd = SGDRegressor()  # 通过梯度下降求解
    sgd.fit(x_train, y_train)

    # 预测结果
    y_predict_sgd = sgd.predict(x_test)
    y_predict_sgd_inverse = std2.inverse_transform(y_predict_sgd)  # 反归一化
    print("sgd预测结果：", y_predict_sgd_inverse)

    # 均方误差
    error_sgd = mean_squared_error(y_test_ori, y_predict_sgd_inverse)
    print("sgd的均分误差：", error_sgd)


if __name__ == "__main__":
    linear()
