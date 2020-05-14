"""
特征工程：
    特征抽取：
        字典特征抽取
        文本特征抽取
    特征预处理：
        归一化
        标准化
        缺失值处理
    特征降维：
        过滤式
        主成分分析
"""

import numpy as np
import pandas as pd


# ndarray与dataframe之间的相互转换
def nd_da():
    data = [[1, 2], [4, 5], [7, 8]]
    print(type(data))
    print("列表：", data)

    # 列表转ndarray
    nd_data = np.array(data)
    print(type(nd_data))
    print("ndarray：", nd_data)

    # ndarray转dataframe
    da_data = pd.DataFrame(nd_data)
    print(type(da_data))
    da_data.columns = ["a", "b"]
    da_data.index = ["A", "B", "C"]
    print("DataFrame:", da_data)

    # dataframe转ndarray
    np_data = np.array(da_data)
    print(type(np_data))
    print(np_data)


# 字典特征抽取：针对特征值是非数值型的特征,进行one_hot编码
from sklearn.feature_extraction import DictVectorizer
def dictvec():
    data = [["北京", 12], ["上海", 50], ["深圳", 100], ["宣城", 1000]]
    data = np.array(data)  # 将列表转成numpy.ndarray
    data = pd.DataFrame(data)  # 将ndarray转成dataframe
    data.columns = ["city", "people"]
    print("dataframe：", data)
    print(type(data))
    dict = DictVectorizer(sparse=False)
    result = dict.fit_transform(data.to_dict(orient="records"))  # 必须这样传，将dataframe转成字典
    print("字典特征抽取之后的结果：", result)
    print(dict.get_feature_names())


# 文本特征抽取：针对特征值是文本的情况
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def text():
    data = ["life is short, I like python", "life is too long, I dislike python"]
    # 统计次数
    cv = CountVectorizer()
    result = cv.fit_transform(data)  # 默认返回稀疏矩阵
    result = result.toarray()  # 将稀疏矩阵转成密集矩阵
    print(result)
    print(type(result))
    print(cv.get_feature_names())
    # 统计重要性
    tf = TfidfVectorizer()
    result_tf = tf.fit_transform(data)
    result_tf = result_tf.toarray()
    print(result_tf)
    print(type(result_tf))
    print(tf.get_feature_names())


# 特征预处理
# 缺失值处理
def missing_value():
    data1 = pd.DataFrame({"一班": [90, 80, 66, 75, 99, 55, 76, 78, 98, None, 90],
                          "二班": [75, 98, 100, None, 77, 45, None, 66, 56, 80, 57],
                          "三班": [45, 89, 77, 67, 65, 100, None, 75, 64, 88, 99],
                          "四班": [45, 89, 77, 67, 65, 100, 45, 75, 64, 88, 99]})

    data2 = pd.DataFrame({"一班": [90, 80, 66, 75, 99, 55, 76, 78, 98, np.nan, 90],
                          "二班": [75, 98, 100, np.nan, 77, 45, np.nan, 66, 56, 80, 57],
                          "三班": [45, 89, 77, 67, 65, 100, np.nan, 75, 64, 88, 99],
                          "四班": [45, 89, 77, 67, 65, 100, 45, 75, 64, 88, 99]})

    data3 = pd.DataFrame({"一班": [90, 80, 66, 75, 99, 55, 76, 78, 98, "null", 90],
                          "二班": [75, 98, 100, "null", 77, 45, "null", 66, 56, 80, 57],
                          "三班": [45, 89, 77, 67, 65, 100, "null", 75, 64, 88, 99],
                          "四班": [45, 89, 77, 67, 65, 100, 45, 75, 64, 88, 99]})
    # 缺失值是None、np.nan都是可以识别出来的，打印的时候显示NaN，但是其他类型的缺失值是无法识别的
    # print(data1)
    # print(data2)
    # print(data3)
    # 如果遇到的是data3这种类型的缺失值，那么首先要用np.nan替换掉缺失值
    data3.replace("null", np.nan, inplace=True)

    # 判断有没有缺失值
    print(pd.isnull(data3).any())

    # 处理缺失值
    # 删除
    # data3.dropna(axis=0, how="any", inplace=True)
    # print(data3)
    # 填充
    data3.fillna(data3.mean(), inplace=True)
    print(data3)

# 归一化
from sklearn.preprocessing import MinMaxScaler
def min_max():
    data = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    mm = MinMaxScaler(feature_range=(0, 1))
    result = mm.fit_transform(data)
    print(result)

# 标准化
from sklearn.preprocessing import StandardScaler
def standard():
    data = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    std = StandardScaler()
    result = std.fit_transform(data)
    print(result)


# 特征抽取：过滤式
from sklearn.feature_selection import VarianceThreshold
def var():
    data = [[1, 2, 3],[1, 4, 5],[1, 7, 8]]
    v = VarianceThreshold(threshold=0)
    result = v.fit_transform(data)
    print(result)


# 特征降维：PCA
from sklearn.decomposition import PCA
def pca():
    data = [[1, 2, 4, 5], [4, 5, 4, 2], [2, 4, 1, 4]]
    p = PCA(n_components=0.95)
    result = p.fit_transform(data)
    print(result)


if __name__ == "__main__":
    nd_da()
    dictvec()
    text()
    missing_value()
    min_max()
    standard()
    var()
    pca()


