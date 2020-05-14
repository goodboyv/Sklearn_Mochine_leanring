from sklearn.datasets import load_iris, load_boston, fetch_20newsgroups


# 加载鸢尾花数据集（分类数据集）
def iris_datasets():
    iris = load_iris()
    feature = iris.data  # 获取特征值
    target = iris.target  # 获取目标值
    feature_names = iris.feature_names  # 获取特征名称
    target_names = iris.target_names  # 获取目标值名称
    print("特征值名称", feature_names)
    print("特征值", feature)
    print("目标值名称", target_names)
    print("目标值", target)
    print("数据集的描述信息", iris.DESCR)


# 获取波士顿房价数据集
def boston_datasets():
    boston = load_boston()
    feature = boston.data
    target = boston.target
    feature_names = boston.feature_names
    #target_names = boston.target_names
    print("特征值名称", feature_names)
    print("特征值", feature)
    #print("目标值名称", target_names)
    print("目标值", target)
    print("数据集的描述信息", boston.DESCR)


# 获取20newsgroups数据集
def newsgroups():
    news = fetch_20newsgroups()
    feature = news.data
    target = news.target
    #feature_names = news.feature_names
    target_names = news.target_names
    #print("特征值名称", feature_names)
    print("特征值", feature)
    print("目标值名称", target_names)
    print("目标值", target)
    print("数据集的描述信息", news.DESCR)

if __name__ == "__main__":
    #iris_datasets()
    #boston_datasets()
    newsgroups()
