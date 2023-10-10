import numpy as np
import pandas as pd
from plotTree import *
import sys

class DecisionTree():
    def __init__(self, criteria="EntropyCriteria"):
        assert criteria in ["EntropyCriteria","GeniCriteria"],"criteria should be EntropyCriteria or GeniCriteria "
        self.criteria = criteria
        self.calculate_criteria = getattr(sys.modules[__name__], criteria)()

    def fit(self,X ,y):
        self.X = X
        self.y = y
        self.data = pd.concat([X, y], axis=1)
        # 所有特征的所有属性值
        self.column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
        self.n_features = X.shape[1]
        self.tree = self.built_tree(self.data)

    def predict(self, X):
        pass

    def built_tree(self,data):
        """
        :param data:
        :return:
        """
        featlist = list(data.columns)  # 提取出数据集所有的列
        classlist = data.iloc[:, -1].value_counts()  # 获取最后一列类标签

        # 当前属性集为空，或是所有样本在所有属性上取值相同，无法划分;
        # 判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
        if data.shape[1] == 1 or classlist[0] == data.shape[0]:
            return classlist.index[0]  # 如果是，返回类标签

        axis = self.best_split(data)  # 确定出当前最佳切分列的索引

        best_feature = featlist[axis]  # 获取该索引对应的特征
        myTree = {best_feature: {}}  # 采用字典嵌套的方式存储树信息
        del featlist[axis]  # 删除当前特征

        valuelist = pd.unique(data.iloc[:, axis])  # 提取最佳切分列所有属性值

        if len(valuelist) != len(self.column_count[best_feature]):
            no_exist_attrs = set(self.column_count[best_feature]) - set(valuelist)  # 少的那些特征
            for no_attr in no_exist_attrs:
                myTree[best_feature][no_attr] = self.get_most_label(data)

        for value in valuelist:  # 对每一个属性值递归建树
            myTree[best_feature][value] = self.built_tree(self.split(data, axis, value))

        return myTree

    def get_most_label(self,data):
        data_label = data.iloc[:, -1]
        label_sort = data_label.value_counts(sort=True)
        return label_sort.keys()[0]

    def split(self,data,axis,value):
        col = data.columns[axis]
        split_data = data.loc[data[col] == value, :].drop(col, axis=1)
        return split_data

    def best_split(self, data):
        print(f"当前处理的属性有：{data.columns}")

        base_criteria = self.calculate_criteria(data)  # 计算原始熵
        bestGain = 0  # 初始化信息增益
        axis = -1  # 初始化最佳切分列，标签列
        for i in range(data.shape[1] - 1):  # 对特征的每一列进行循环
            levels = data.iloc[:, i].value_counts().index  # 提取出当前列的所有取值
            criterias = 0  # 初始化子节点的信息熵
            for j in levels:  # 对当前列的每一个取值进行循环
                subSet = data[data.iloc[:, i] == j]  # 某一个子节点的dataframe
                criteria = self.calculate_criteria(subSet)  # 计算某一个子节点的信息熵
                criterias += (subSet.shape[0] / data.shape[0]) * criteria  # 计算当前列的信息熵

            print(f'{data.columns[i]}  列的信息熵为{criterias}')
            infoGain = base_criteria - criterias  # 计算当前列的信息增益
            print(f'{data.columns[i]}  列的信息增益为{infoGain}')
            if (infoGain > bestGain):
                bestGain = infoGain  # 选择最大信息增益
                axis = i  # 最大信息增益所在列的索引
        print(f"我们选择 {data.columns[axis]} 属性作为最佳划分")
        print(f"该属性中有 {[x for x in set(data.iloc[:,axis])]}\n")
        return axis


class EntropyCriteria():

    def __call__(self, data):
        n = data.shape[0]  # 数据集总行数
        iset = data.iloc[:, -1].value_counts()  # 标签的所有类别
        p = iset / n  # 每一类标签所占比
        ent = (-p * np.log2(p)).sum()  # 计算信息熵
        return ent

class GeniCriteria():

    def __call__(self, data):
        n = data.shape[0]  # 数据集总行数
        iset = data.iloc[:, -1].value_counts()  # 标签的所有类别
        p = iset / n  # 每一类标签所占比
        geni = 1 - (p*p).sum()
        return geni


if __name__ == '__main__':
    data = pd.read_csv("./data/watermelon2delete.csv")


    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    model = DecisionTree(criteria="GeniCriteria")
    model.fit(X,y)
    tree = model.tree
    createPlot(tree)
