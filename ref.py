
# > https://blog.csdn.net/yoggieCDA/article/details/92832367

import numpy as np
import pandas as pd
from plotTree import *

#创建数据集
def createDataSet():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet

def calEnt(dataSet):
    """
    函数功能：计算香农熵
    参数说明：
        dataSet：原始数据集
    返回：
    	ent:香农熵的值
    """
    n = dataSet.shape[0]                             #数据集总行数
    iset = dataSet.iloc[:,-1].value_counts()         #标签的所有类别
    p = iset/n                                       #每一类标签所占比
    ent = (-p*np.log2(p)).sum()                      #计算信息熵
    return ent


# 选择最优的属性进行切分
def bestSplit(dataSet):
    """
    函数功能：根据信息增益选择出最佳数据集切分的列
    参数说明：
    	dataSet：原始数据集
    返回：
    	axis:数据集最佳切分列的索引
    """
    baseEnt = calEnt(dataSet)                                #计算原始熵
    bestGain = 0                                             #初始化信息增益
    axis = -1                                                #初始化最佳切分列，标签列
    for i in range(dataSet.shape[1]-1):                      #对特征的每一列进行循环
        levels= dataSet.iloc[:,i].value_counts().index       #提取出当前列的所有取值
        ents = 0      										 #初始化子节点的信息熵
        for j in levels:									 #对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:,i]==j]         #某一个子节点的dataframe
            ent = calEnt(childSet)							 #计算某一个子节点的信息熵
            ents += (childSet.shape[0]/dataSet.shape[0])*ent #计算当前列的信息熵
        print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt-ents								 #计算当前列的信息增益
        print(f'第{i}列的信息增益为{infoGain}')
        if (infoGain > bestGain):
            bestGain = infoGain                              #选择最大信息增益
            axis = i                                         #最大信息增益所在列的索引
    return axis


def mySplit(dataSet,axis,value):
    """
    函数功能：按照给定的列划分数据集
    参数说明：
    	dataSet：原始数据集
    	axis：指定的列索引
    	value：指定的属性值
    返回：
    	redataSet：按照指定列索引和属性值切分后的数据集
    """
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col]==value,:].drop(col,axis=1)
    return redataSet



def createTree(dataSet):
    """
    函数功能：基于最大信息增益切分数据集，递归构建决策树
    参数说明：
    	dataSet：原始数据集（最后一列是标签）
    返回：
    	myTree：字典形式的树
    """
    featlist = list(dataSet.columns)                          #提取出数据集所有的列
    classlist = dataSet.iloc[:,-1].value_counts()             #获取最后一列类标签
    #判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if classlist[0]==dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]                             #如果是，返回类标签
    axis = bestSplit(dataSet)                                 #确定出当前最佳切分列的索引
    bestfeat = featlist[axis]                                 #获取该索引对应的特征
    myTree = {bestfeat:{}}                                    #采用字典嵌套的方式存储树信息
    del featlist[axis]                                        #删除当前特征
    valuelist = set(dataSet.iloc[:,axis])                     #提取最佳切分列所有属性值
    for value in valuelist:	                                  #对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
    return myTree


if __name__ == '__main__':
    # dataSet = createDataSet()
    dataSet = pd.read_csv("./data/watermelon2.csv")
    dataSet = dataSet.iloc[:,1:]
    ent = calEnt(dataSet)
    myTree = createTree(dataSet)
    # 保存决策树模型
    createPlot(myTree)