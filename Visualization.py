# > https://mp.weixin.qq.com/s/1mdaKvMSlhA87KlpPBk8Mw
# > https://mp.weixin.qq.com/s/74ehblIzwe4rmCM6K-ynEA
# 安装完Graphviz需要重启，才能有用


"""
生成的图中参数的意义
    gini：节点的基尼不纯度。当沿着树向下移动时，平均加权的基尼不纯度必须降低。
    samples：节点中观察的数量。
    value：每一类别中样本的数量。比如，顶部节点中有 2 个样本属于类别 0，有 4 个样本属于类别 1。
    class：节点中大多数点的类别（持平时默认为 0）。在叶节点中，这是该节点中所有样本的预测结果。
"""

#导入相应的包
import pydotplus
from sklearn.datasets import *
from dtreeviz.trees import *
from sklearn import tree
import pandas as pd
from sklearn.datasets import load_iris
from graphviz import Digraph


#创建数据集
def createDataSet():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet



def createWatermeloneDataset(source_path,target_path):

    # 创建属于西瓜的数据集
    dataSet_df = pd.read_csv(source_path, header=0)
    dataSet = np.array(dataSet_df)
    # 遍历dataSet
    for index, value in enumerate(dataSet):
        dataSet[index] = list(dataSet[index])
        for index2, value2 in enumerate(dataSet[index]):
            # 判断如果是字符串，去除前后的字符
            if (type(dataSet[index][index2]) == str):
                dataSet[index][index2] = dataSet[index][index2].strip()
            # 如果不是字符串，转换为字符串
            else:
                dataSet[index][index2] = str(dataSet[index][index2])
    # labels
    labels = list(np.array(dataSet_df.columns))

    return dataSet, labels


def graphViz_sample(csv_path=None):
    clf = tree.DecisionTreeClassifier()

    if csv_path is None:
        iris = load_iris()
        clf = clf.fit(iris.data, iris.target)
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=iris.feature_names,
                                        class_names=iris.target_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
    else:
        data = pd.read_csv(csv_path,header=0)
        col = list(data.head())
        class_name = data.iloc[:,-1].unique()
        X = data.iloc[:, 1:-1]
        y = data.iloc[:, -1]
        clf = clf.fit(X,y)
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=col,
                                        class_names=class_name,
                                        filled=True,rounded=True,
                                        special_characters=True)

        dot_data = None

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("Decision Tree.pdf")




def dtreeviz_sample():
    iris = load_iris()
    classifier = tree.DecisionTreeClassifier()
    viz = model(classifier,
                   iris.data,
                   iris.target,
                   tree_index=1,
                   target_name="variety",
                   feature_names=iris.feature_names,
                   class_names=["setosa","vericolor","virginica"])
    viz.view()



# 可视化用字典存储的树 还存在一些问题.
def visDictTree(tree=None):
    """
    tree 的结构如下：{'no surfacing': {'0': 'no', '1': {'flippers': {0: 'no', 1: 'yes'}}}}
        0 --> 不符合条件  1  -->  符合判定条件
    """

    if tree is None:
        tree = {'no surfacing': {'0': 'no', '1': {'flippers': {'0': 'no', '1': 'yes'}}}}

    # 创建一个Digraph对象，设置一些全局属性
    dot = Digraph(comment="Decision Tree", format="png", engine="dot")
    dot.attr(rankdir="TB")  # 设置布局方向为从上到下

    # 定义一个递归函数来遍历树的结构并添加节点和边
    def add_nodes_edges(dot, tree, parent_node=None, parent_label=None):

        # 根节点
        if parent_node is None:
            nodel_label = [key for key in tree.keys()][0]
            child = tree[nodel_label]
            dot.node(nodel_label,shape="ellipse", color="lightblue2", style="filled")
            add_nodes_edges(dot, child, nodel_label, nodel_label)

        else:
            for label, child in tree.items():
                # 创建节点标签，包括额外信息
                if isinstance(child,str):
                    node_label = child
                    edge_label = label
                    dot.node(node_label, shape="ellipse", color="lightblue2", style="filled")
                    dot.edge(parent_label, node_label, label=edge_label)
                else:
                    edge_label = label
                    add_nodes_edges(dot, child, parent_node, edge_label)


    # 调用函数开始添加节点和边
    add_nodes_edges(dot, tree)

    # 渲染和保存图
    dot.render("decision_tree_visualization", view=True)




if __name__ == '__main__':
    graphViz_sample()





