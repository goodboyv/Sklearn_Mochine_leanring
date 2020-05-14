"""
    通过树形结构来实现分类的一种算法，关键在于如何选择最优属性
    通常用三种方式：信息增益（ID3）、增益率（C4.5）、基尼系数（CART）
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split


def de_tree():
    # 加载数据
    titan = pd.read_csv("E:/Desktop/机器学习_新/数据集/泰坦尼克数据集/train.csv")
    # print(titan.shape)
    # pd.set_option("display.max_columns", 100)  # 把dataframe中省略的部分显示出来
    # print(titan.head(5))

    # 构造特征值和目标值
    feature = titan[["Pclass", "Age", "Fare", "Sex"]]
    target = titan["Survived"]

    # 特征预处理
    # 查看有没有缺失值
    print(pd.isnull(feature).any())
    # 填充缺失值
    Age = feature.pop("Age")  # 取出，意思是取出来之后删除原来的
    Age = Age.fillna(Age.mean())
    # print(feature)
    # feature.drop("Age", axis=1, inplace=True)  # 删除一列
    feature.insert(0, "Age", Age)
    # print(pd.isnull(feature).any())

    # 字典特征抽取
    dv = DictVectorizer()
    feature = dv.fit_transform(feature.to_dict(orient="records"))
    feature = feature.toarray()
    print(feature)
    print(dv.get_feature_names())

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 建立模型
    tree = DecisionTreeClassifier(max_depth=5)

    # 训练
    tree.fit(x_train, y_train)

    # 验证
    score = tree.score(x_val, y_val)
    print("在验证集上的得分：", score)

    # 预测
    score_test = tree.score(x_test, y_test)
    print("在测试集上的得分：", score_test)
    predict = tree.predict(x_test)
    print("测试结果：", predict)

    # 保存树结构
    export_graphviz(tree, out_file="E:/Desktop/开题报告/tree.dot", feature_names=['Age', 'Fare', 'Pclass', 'Sex=female', 'Sex=male'])

    # 将保存的dot文件转成png文件，查看树结构
    # dot -Tpng tree.dot -o tree.png


if __name__ == "__main__":
    de_tree()