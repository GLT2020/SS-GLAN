import networkx as nx
import json
import os
import numpy as np
from os.path import join
import dgl
from dgl.data import DGLDataset
from tqdm import tqdm
import pickle

import matplotlib
import matplotlib.pyplot as plt


import fasttext
import fasttext.util

# fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('../cc.en.300.bin')

ROOT = '../data/ge-sc-data/source_code'
# bug_type = {'access_control': 57*2, 'arithmetic': 60*2, 'denial_of_service': 46*2,
#           'front_running': 44*2, 'reentrancy': 71*2, 'time_manipulation': 50*2,
#           'unchecked_low_level_calls': 95*2}


def TyepEmbedding(typelist):
    type_vectors = []
    # 将类别词转换为向量
    for line_types in typelist:
        line_type_vectors = [ft.get_word_vector(type) for type in line_types]
        # 计算平均向量
        average_vector = sum(line_type_vectors) / len(line_type_vectors)
        type_vectors.append(average_vector)

    return type_vectors

def Nx_graphToDGL():
    bug_type = {'access_control': 57 * 2}
    GRAPHPath = 'all_mixed_graph'
    for bug, counter in bug_type.items():
        vectors_file = f'../data/ge-sc-data/source_code/{bug}/LineVectorList.json'
        types_file = f'../data/ge-sc-data/source_code/{bug}/typeList.json'
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)
        with open(types_file, 'r') as f:
            types = json.load((f))

        source = f'{ROOT}/{bug}/{GRAPHPath}'
        output = f'{ROOT}/{bug}/Mixed_grap_DGL.json'
        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.gpickle')]

        files_vector_dict = {}
        graph_dict = {}
        for files in tqdm(smart_contracts):
            file_name = files.split('/')[-1].split('.')
            file_name = file_name[0] + '.' + file_name[1]

            # 读取对应文件的向量和类别
            contract_vectors = vectors[file_name]
            contract_types = types[file_name]
            type_list = types[file_name]
            type_vectors = TyepEmbedding(type_list)
            line_type_vectors = np.concatenate((contract_vectors, type_vectors), axis=1)  # 拼接行向量和类型向量

            # 读取 gpickle 文件并转换为 NetworkX 图
            nx_graph = nx.read_gpickle(f'{source}/{file_name}.gpickle')

            # 遍历每个节点，计算节点特征
            node_features_dict = {}
            for node, data in nx_graph.nodes(data=True):
                # 获取 node_source_code_lines 属性
                source_code_lines = data['node_source_code_lines']
                line_features = line_type_vectors[source_code_lines]

                # 计算节点的特征（多行特征的平均值）
                node_feature = np.mean(line_features, axis=0)

                # 存储节点特征
                node_features_dict[node] = node_feature

            # 将 NetworkX 图转换为 DGL 图
            dgl_graph = dgl.from_networkx(nx_graph)

            # 获取节点特征矩阵
            node_features = [node_features_dict[i] for i in range(dgl_graph.num_nodes())]

            # 将特征矩阵赋值给 DGL 图的节点特征
            dgl_graph.ndata['feat'] = node_features

            # 创建一个字典来存储文件名和图特征
            graph_dict = {
                file_name: {
                    "graph": dgl_graph,
                    "features": node_features
                }
            }

        # 将图特征保存到文件中
        with open('graph_data.pkl', 'wb') as f:
            pickle.dump(graph_dict, f)

def RepairNx_Graph(file_path):
    # 读取 gpickle 文件并转换为 NetworkX 图
    nx_graph = nx.read_gpickle(f'{file_path}.gpickle')
    for node, data in nx_graph.nodes(data=True):
        # 获取 node_source_code_lines 属性
        source_code_lines = data['node_source_code_lines']
        if 306 in source_code_lines:
            del(data['node_source_code_lines'][-1])

    nx.write_gpickle(nx_graph, f'{file_path}.gpickle')


if __name__ == '__main__':
    bug_type = 'access_control'
    file_path = f'{ROOT}/{bug_type}/all_cfg_graph/buggy_16.sol'
    RepairNx_Graph(file_path)