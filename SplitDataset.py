
import json
import torch
import numpy as np
import networkx as nx
import dgl
import pickle
import joblib
import os

import fasttext
import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

from sklearn.decomposition import PCA

from parser1 import parameter_parser
args = parameter_parser()


# 加载训练好的PCA模型
with open(f'./pca_pkl/PCA_{args.type}_vector.pkl', 'rb') as f:
    PCA_vector = pickle.load(f)




def TyepEmbedding(typelist):
    type_vectors = []
    # 将类别词转换为向量
    for line_types in typelist:
        line_type_vectors = [ft.get_word_vector(type) for type in line_types]
        # 计算平均向量
        average_vector = sum(line_type_vectors) / len(line_type_vectors)
        type_vectors.append(average_vector)

    return type_vectors


def Split_dataset_vectortype(vectors_file, labels_file, types_file):
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []
    for file_item in raw_labels:
        contract_name = file_item['name']
        # print(f'contract_name{contract_name}')
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]
        type_list = types[contract_name]
        type_vectors = TyepEmbedding(type_list)
        line_type_vectors = np.concatenate((vectors_list, type_vectors), axis=1)  # 拼接行向量和类型向量

        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0
        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities


# 使用pca压缩vector
def Split_dataset_vectortype_pca(vectors_file, labels_file, types_file):
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []
    for file_item in raw_labels:
        contract_name = file_item['name']
        # print(f'contract_name{contract_name}')
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]


        vectors_list = PCA_vector.transform(vectors_list)

        type_list = types[contract_name]
        type_vectors = TyepEmbedding(type_list)
        line_type_vectors = np.concatenate((vectors_list, type_vectors[: len(vectors_list)]), axis=1)  # 拼接行向量和类型向量

        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0
        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities


def Split_dataset_vector(vectors_file, labels_file):
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []
    for file_item in raw_labels:
        contract_name = file_item['name']
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]
        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0
        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, vectors_list, line_labels, flag_labels))
        else:
            contracts_without_vulnerabilities.append((contract_name, vectors_list, line_labels, flag_labels))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities

def Split_dataset_type(types_file, labels_file ):

    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []
    for file_item in raw_labels:
        contract_name = file_item['name']
        # print(f'contract_name{contract_name}')
        labels = file_item['vulnerabilities'][0]['lines']
        type_list = types[contract_name]

        type_vectors = TyepEmbedding(type_list)
        line_type_vectors = type_vectors

        line_labels = [1 if i + 1 in labels else 0 for i in range(len(type_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0
        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities






def Split_dataset_graphvector(vectors_file, labels_file):
    # 加载向量和标签
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)


    ROOT = './data/ge-sc-data/source_code'
    GRAPHPath = 'all_mixed_graph'

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []

    for file_item in raw_labels:
        contract_name = file_item['name']
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]

        line_type_vectors = np.array(vectors_list)
        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0

        # 读取 gpickle 文件并转换为 NetworkX 图
        nx_graph = nx.read_gpickle(f'{ROOT}/{args.type}/{GRAPHPath}/{contract_name}.gpickle')

        # 将 NetworkX 图转换为 DGL 图
        dgl_graph = dgl.from_networkx(nx_graph)

        # print(f'contract_name:{contract_name}')
        # 初始化节点特征,遍历每个节点，计算节点特征
        node_features_dict = {}
        for node, data in nx_graph.nodes(data=True):
            # 获取 node_source_code_lines 属性
            source_code_lines = data['node_source_code_lines']

            # 生成的图会出现超出index的情况。需要将超出的部分删掉
            source_code_lines =  [x for x in source_code_lines if x <= len(line_type_vectors)]

            # 向量index从0开始
            source_code_lines = [(item - 1) for item in source_code_lines]
            line_features = line_type_vectors[source_code_lines]
            # 计算节点的特征（多行特征的平均值）
            node_feature = np.mean(line_features, axis=0)
            # 存储节点特征
            node_features_dict[node] = node_feature

        # 获取节点特征矩阵,确保节点特征顺序与dgl的节点顺序一致
        # node_features = [node_features_dict[i] for i in range(dgl_graph.num_nodes())]
        node_features = np.array([node_features_dict[node] for node in nx_graph.nodes])
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        # 将特征矩阵赋值给 DGL 图的节点特征
        dgl_graph.ndata['feat'] = node_features_tensor
        dgl_graph = dgl.add_self_loop(dgl_graph)

        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities


def Split_dataset_graphtype(types_file, labels_file):
    # 加载向量和标签
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))


    ROOT = './data/ge-sc-data/source_code'
    GRAPHPath = 'all_mixed_graph'

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []

    for file_item in raw_labels:
        contract_name = file_item['name']
        labels = file_item['vulnerabilities'][0]['lines']
        type_list = types[contract_name]

        type_vectors = TyepEmbedding(type_list)
        line_type_vectors = np.array(type_vectors)

        line_labels = [1 if i + 1 in labels else 0 for i in range(len(type_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0

        # 读取 gpickle 文件并转换为 NetworkX 图
        nx_graph = nx.read_gpickle(f'{ROOT}/{args.type}/{GRAPHPath}/{contract_name}.gpickle')

        # 将 NetworkX 图转换为 DGL 图
        dgl_graph = dgl.from_networkx(nx_graph)

        # print(f'contract_name:{contract_name}')
        # 初始化节点特征,遍历每个节点，计算节点特征
        node_features_dict = {}
        for node, data in nx_graph.nodes(data=True):
            # 获取 node_source_code_lines 属性
            source_code_lines = data['node_source_code_lines']

            # 生成的图会出现超出index的情况。需要将超出的部分删掉
            source_code_lines =  [x for x in source_code_lines if x <= len(line_type_vectors)]

            # 向量index从0开始
            source_code_lines = [(item - 1) for item in source_code_lines]
            line_features = line_type_vectors[source_code_lines]
            # 计算节点的特征（多行特征的平均值）
            node_feature = np.mean(line_features, axis=0)
            # 存储节点特征
            node_features_dict[node] = node_feature

        # 获取节点特征矩阵,确保节点特征顺序与dgl的节点顺序一致
        # node_features = [node_features_dict[i] for i in range(dgl_graph.num_nodes())]
        node_features = np.array([node_features_dict[node] for node in nx_graph.nodes])
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        # 将特征矩阵赋值给 DGL 图的节点特征
        dgl_graph.ndata['feat'] = node_features_tensor
        dgl_graph = dgl.add_self_loop(dgl_graph)

        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities


def Split_dataset_graphvectortype(vectors_file, labels_file, types_file):
    # 加载向量和标签
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))

    ROOT = './data/ge-sc-data/source_code'
    GRAPHPath = 'all_mixed_graph'

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []

    for file_item in raw_labels:
        contract_name = file_item['name']
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]
        type_list = types[contract_name]
        type_vectors = TyepEmbedding(type_list)
        line_type_vectors = np.concatenate((vectors_list, type_vectors), axis=1)  # 拼接行向量和类型向量
        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0

        # 读取 gpickle 文件并转换为 NetworkX 图
        nx_graph = nx.read_gpickle(f'{ROOT}/{args.type}/{GRAPHPath}/{contract_name}.gpickle')

        # 将 NetworkX 图转换为 DGL 图
        dgl_graph = dgl.from_networkx(nx_graph)

        # print(f'contract_name:{contract_name}')
        # 初始化节点特征,遍历每个节点，计算节点特征
        node_features_dict = {}
        for node, data in nx_graph.nodes(data=True):
            # 获取 node_source_code_lines 属性
            source_code_lines = data['node_source_code_lines']

            # 生成的图会出现超出index的情况。需要将超出的部分删掉
            source_code_lines =  [x for x in source_code_lines if x <= len(line_type_vectors)]

            # 向量index从0开始
            source_code_lines = [(item - 1) for item in source_code_lines]
            line_features = line_type_vectors[source_code_lines]
            # 计算节点的特征（多行特征的平均值）
            node_feature = np.mean(line_features, axis=0)
            # 存储节点特征
            node_features_dict[node] = node_feature

        # 获取节点特征矩阵,确保节点特征顺序与dgl的节点顺序一致
        # node_features = [node_features_dict[i] for i in range(dgl_graph.num_nodes())]
        node_features = np.array([node_features_dict[node] for node in nx_graph.nodes])
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        # 将特征矩阵赋值给 DGL 图的节点特征
        dgl_graph.ndata['feat'] = node_features_tensor
        dgl_graph = dgl.add_self_loop(dgl_graph)

        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities

def Split_dataset_graphvectortype_pca(vectors_file, labels_file, types_file):
    # 加载向量和标签
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))

    ROOT = './data/ge-sc-data/source_code'
    GRAPHPath = 'all_mixed_graph'

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []

    for file_item in raw_labels:
        contract_name = file_item['name']
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]

        vectors_list = PCA_vector.transform(vectors_list)

        type_list = types[contract_name]
        type_vectors = TyepEmbedding(type_list)
        line_type_vectors = np.concatenate((vectors_list, type_vectors[: len(vectors_list)]), axis=1)  # 拼接行向量和类型向量
        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0

        # 读取 gpickle 文件并转换为 NetworkX 图
        nx_graph = nx.read_gpickle(f'{ROOT}/{args.type}/{GRAPHPath}/{contract_name}.gpickle')

        # 将 NetworkX 图转换为 DGL 图
        dgl_graph = dgl.from_networkx(nx_graph)

        # print(f'contract_name:{contract_name}')
        # 初始化节点特征,遍历每个节点，计算节点特征
        node_features_dict = {}
        for node, data in nx_graph.nodes(data=True):
            # 获取 node_source_code_lines 属性
            source_code_lines = data['node_source_code_lines']

            # 生成的图会出现超出index的情况。需要将超出的部分删掉
            source_code_lines =  [x for x in source_code_lines if x <= len(line_type_vectors)]

            # 向量index从0开始
            source_code_lines = [(item - 1) for item in source_code_lines]
            line_features = line_type_vectors[source_code_lines]
            # 计算节点的特征（多行特征的平均值）
            node_feature = np.mean(line_features, axis=0)
            # 存储节点特征
            node_features_dict[node] = node_feature

        # 获取节点特征矩阵,确保节点特征顺序与dgl的节点顺序一致
        # node_features = [node_features_dict[i] for i in range(dgl_graph.num_nodes())]
        node_features = np.array([node_features_dict[node] for node in nx_graph.nodes])
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        # 将特征矩阵赋值给 DGL 图的节点特征
        dgl_graph.ndata['feat'] = node_features_tensor
        dgl_graph = dgl.add_self_loop(dgl_graph)

        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities

# 将vector和type特征相加融合。即最终特征维度为300
def Split_dataset_graphvectortype_pca_add(vectors_file, labels_file, types_file):
    # 加载向量和标签
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    with open(labels_file, 'r') as f:
        raw_labels = json.load(f)
    with open(types_file, 'r') as f:
        types = json.load((f))

    ROOT = './data/ge-sc-data/source_code'
    GRAPHPath = 'all_mixed_graph'

    # 按有漏洞和无漏洞划分数据
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []

    for file_item in raw_labels:
        contract_name = file_item['name']
        labels = file_item['vulnerabilities'][0]['lines']
        vectors_list = vectors[contract_name]

        vectors_list = PCA_vector.transform(vectors_list)

        type_list = types[contract_name]
        type_vectors = TyepEmbedding(type_list)
        # vector和type相加
        line_type_vectors = vectors_list + type_vectors[: len(vectors_list)]  # 拼接行向量和类型向量
        line_labels = [1 if i + 1 in labels else 0 for i in range(len(vectors_list))]  # 将标签转为:[0,0,0,1,1,0,1]的形式
        flag_labels = 1 if len(labels) > 0 else 0

        # 读取 gpickle 文件并转换为 NetworkX 图
        nx_graph = nx.read_gpickle(f'{ROOT}/{args.type}/{GRAPHPath}/{contract_name}.gpickle')

        # 将 NetworkX 图转换为 DGL 图
        dgl_graph = dgl.from_networkx(nx_graph)

        # print(f'contract_name:{contract_name}')
        # 初始化节点特征,遍历每个节点，计算节点特征
        node_features_dict = {}
        for node, data in nx_graph.nodes(data=True):
            # 获取 node_source_code_lines 属性
            source_code_lines = data['node_source_code_lines']

            # 生成的图会出现超出index的情况。需要将超出的部分删掉
            source_code_lines =  [x for x in source_code_lines if x <= len(line_type_vectors)]

            # 向量index从0开始
            source_code_lines = [(item - 1) for item in source_code_lines]
            line_features = line_type_vectors[source_code_lines]
            # 计算节点的特征（多行特征的平均值）
            node_feature = np.mean(line_features, axis=0)
            # 存储节点特征
            node_features_dict[node] = node_feature

        # 获取节点特征矩阵,确保节点特征顺序与dgl的节点顺序一致
        # node_features = [node_features_dict[i] for i in range(dgl_graph.num_nodes())]
        node_features = np.array([node_features_dict[node] for node in nx_graph.nodes])
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        # 将特征矩阵赋值给 DGL 图的节点特征
        dgl_graph.ndata['feat'] = node_features_tensor
        dgl_graph = dgl.add_self_loop(dgl_graph)

        if flag_labels:
            contracts_with_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))
        else:
            contracts_without_vulnerabilities.append((contract_name, line_type_vectors, line_labels, flag_labels, dgl_graph))

    return contracts_with_vulnerabilities, contracts_without_vulnerabilities


if __name__ == '__main__':

   pass