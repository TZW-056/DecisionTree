import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def BinaryEnt(x):
    return -(x*np.log2(x)+(1-x)*np.log2(1-x))

def saveIris():
    iris = load_iris()
    col = list(iris["feature_names"])  # col是列名
    # 在iris数据集中，标签在"data"数组里，标记在"target"数组里
    m1 = pd.DataFrame(iris.data, index=range(150), columns=col)
    m2 = pd.DataFrame(iris.target, index=range(150), columns=["outocme"])

    # 将上述两张DataFrame表连接起来，how是DataFrame参数，可以不写，这里用外连接。不清楚外连接的可以看下SQL语句
    m3 = m1.join(m2, how='outer')

    # to_excel语句转化成excel格式，后缀名为.xls
    m3.to_excel("./iris.xlsx")

if __name__ == '__main__':
    data = pd.read_csv("./data/watermelon2.csv")
    print(data.columns[2])
    data.drop()