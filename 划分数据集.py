from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 将数据集划分成训练集、验证集、测试集
def split_datasets():
    iris = load_iris()
    feature = iris.data
    target = iris.target
    print("特征值：", type(feature))
    print("目标值：", type(target))
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)  # 传的参数必须是numpy.ndarray或者pandas.dataframes，但是必须是传入特征值和目标值，不能一起传入
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    print("训练集：", x_train.shape, y_train.shape)
    print("验证集：", x_val.shape, y_val.shape)
    print("测试集：", x_test.shape, y_test.shape)


if __name__ == "__main__":
    split_datasets()

