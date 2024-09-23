import json
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_bytecode(file):
    info_data = []
    # 混合的数据集按比例切割
    # path = "./result/"+ file +".json"
    # print(path)
    with open(file, "r") as f:
        for line in f:
            try:
                info_data.append(json.loads(line.rstrip('\n')))
            except ValueError:
                print("Skipping invalid line {0}".format(repr(line)))
    return info_data


def get_all_ave(data: list):
    acc = []
    recall = []
    precision = []
    F1 = []
    roc = []
    for i in range(len(data)):
        acc.append(data[i]['accuracy'])
        recall.append(data[i]['recall'])
        precision.append(data[i]['precision'])
        F1.append(data[i]['F1'])
        # roc.append(data[i]['roc'])
    mean_acc = np.mean(acc)  # 平均值
    mean_recall = np.mean(recall)  # 平均值
    mean_precision = np.mean(precision)  # 平均值
    mean_F1 = np.mean(F1)  # 平均值
    # mean_roc = np.mean(roc)

    std_acc = np.std(acc)
    std_recall = np.std(recall)
    std_precision = np.std(precision)
    std_F1 = np.std(F1)
    # std_roc = np.std(roc)

    # print("--------------------------------------")
    print("mean_acc:", mean_acc, ";mean_recall:", mean_recall, ";mean_precision:", mean_precision, ";mean_F1:", mean_F1)
    print("std_acc:", std_acc, ";std_recall:", std_recall, ";std_precision:", std_precision, ";std_F1:", std_F1)
    # print("mean_acc:", mean_acc, ";mean_recall:", mean_recall, ";mean_precision:", mean_precision, ";mean_F1:", mean_F1,
    #       'mean_roc:', mean_roc)
    # print("std_acc:", std_acc, ";std_recall:", std_recall, ";std_precision:", std_precision, ";std_F1:", std_F1,
    #       "std_roc:", std_roc)
    print("--------------------------------------\n")
    return np.array([mean_acc, std_acc, mean_recall, std_recall, mean_precision, std_precision, mean_F1, std_F1])


if __name__ == '__main__':

    types = [
        'access_control',
        'arithmetic',
        'denial_of_service',
        'front_running',
        'reentrancy',
        'time_manipulation',
        'unchecked_low_level_calls'
    ]

    model_types = [
                  # 'transformer',
                  # 'transformer_contract', 'transformerencoder_contract',
                    'transformerencoder',
                  #   'gat_transformerencoder',
                    # 'gcn_transformerencoder',
                  # 'gcn',
                  # 'gat',
                  #   'gat_lstm',
                  #   'lstm'
                  ]
    # file = './result/access_control_vector_transformerencoder_640_300.json'

    DTs = [
        # 'vector', 'type',
        'vectortype',
        # 'graph_vector', 'graph_type',
        # 'graph_vectortype'
    ]
    # DT = 'vectortype'
    # DT = 'graph_vectortype'

    PCA = 'True'
    # PCA = 'False'
    # PCA = 'add'

    CONTRACT = True
    # CONTRACT = False

    EPOCH = 100

    for type in types:
        for key in model_types:
            for DT in DTs:
                if CONTRACT == True:
                    file = f'./print_result/{type}/{DT}_{key}_contract_640_{EPOCH}_{PCA}.json'
                else:
                    file = f'./print_result/{type}/{DT}_{key}_640_{EPOCH}_{PCA}.json'
                print(f'************************************************{type}************************************************')
                print(f'file:{file}')
                print(f'==================={key}====================')
                print(f'------------------{DT}----------------')
                info = read_bytecode(file)
                temp = get_all_ave(info)
