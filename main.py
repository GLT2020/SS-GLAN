import json
import torch
import os
import time


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random

from dataloader.VectorTypeDataloader import VectorTypeDataset, VectorTypeCollate_fn
from dataloader.LineVectorDataloader import LineVectorDataset, LineVectorCollate_fn
from dataloader.LineTypeDataloader import LineTypeDataset, LineTypecollate_fn
from dataloader.GraphVectorTypeDataloader import GraphVectorTypeDataset, GraphVectorTypeCollate_fn

from model.GraphLSTM import GraphLSTMModel

from model.GraphLSTMContract import GraphLSTMContractModel


from SplitDataset import Split_dataset_vectortype, Split_dataset_vector, Split_dataset_type, \
    Split_dataset_graphvectortype, Split_dataset_graphvector,  Split_dataset_graphtype, \
    Split_dataset_vectortype_pca, Split_dataset_graphvectortype_pca, Split_dataset_graphvectortype_pca_add

from parser1 import parameter_parser
args = parameter_parser()

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(42)



# 将测试模型的数据保存下来:name数据集名，model：模型类别
def save_modle_result(test_result):
    dic = {}
    key = ["tp", "fp", "tn", "fn", "accuracy", "precision", "recall",  "F1", "test_loss"]
    for index, value in enumerate(test_result):
        dic[key[index]] = value
    info_json = json.dumps(dic)

    path = f"./print_result/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    with open(path, "a+") as f:
        # pickle.dump(data, my_file)
        f.write(info_json + "\n")



if __name__ == '__main__':
    vectors_file = f'./data/ge-sc-data/source_code/{args.type}/LineVectorList.json'
    labels_file = f'./data/ge-sc-data/source_code/{args.type}/preprocess_all.json'


    types_file = f'./data/ge-sc-data/source_code/{args.type}/typeList.json'

    if args.data_type == 'vectortype':
        if args.pca == 'True':
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_vectortype_pca(vectors_file,
                                                                                                         labels_file, types_file)
        else:
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_vectortype(vectors_file,
                                                                                                     labels_file,
                                                                                                     types_file)
    elif args.data_type == 'vector':
        contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_vector(vectors_file,
                                                                                                     labels_file)
    elif args.data_type == 'type':
        contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_type(types_file,
                                                                                                     labels_file)

    elif args.data_type == 'graph_vectortype':
        if args.pca == 'True':
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvectortype_pca(args.type, vectors_file, labels_file, types_file)
        else:
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvectortype(args.type,vectors_file, labels_file,
                                                                                                          types_file)
    elif args.data_type == 'graph_vector':
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvector(args.type,vectors_file, labels_file)

    elif args.data_type == 'graph_type':
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphtype(args.type,types_file, labels_file)



    # 计算有漏洞行数和无漏洞函数、用于加权权重
    all_vul_labels= [item for sublabel in  contracts_with_vulnerabilities for item in sublabel[2]]
    all_nor_labels= [item for sublabel in  contracts_without_vulnerabilities for item in sublabel[2]]
    all_labels = all_vul_labels+all_nor_labels
    total_samples = len(all_labels)
    # total_samples = [len(item) for sublabel in contracts_with_vulnerabilities for item in sublabel[1]]
    num_with_vulnerabilities = sum(1 for labels in all_labels if labels)  # 漏洞类别数量
    num_without_vulnerabilities = total_samples - num_with_vulnerabilities  # 无漏洞类别数量


    for times in range(1):
        print(f"the {times} times to train the {args.type}")
        # 输出配置
        print(f'model:{args.model};         data_type:{args.data_type};      '
              f'vul_type:{args.type};        epochs:{args.epochs};          model_dim:{args.model_dim}  mode:{args.pca}')
        print(f'all dataset in num_without_vulnerabilities:{num_without_vulnerabilities};       num_with_vulnerabilities：{num_with_vulnerabilities} ')
        class_weight = [1.0, 1.0]

        print(f'class weight:{class_weight}')

        train_with, test_with = train_test_split(contracts_with_vulnerabilities, test_size=0.3)
        train_without, test_without = train_test_split(contracts_without_vulnerabilities, test_size=0.3)

        # 合并训练集和测试集
        train_data = train_with + train_without
        test_data = test_with + test_without

        test_labels = [item for sublabel in test_data for item in sublabel[2]]
        test_total_samples = len(test_labels)
        test_num_with_vulnerabilities = sum(1 for labels in test_labels if labels)  # 漏洞类别数量
        test_num_without_vulnerabilities = test_total_samples - test_num_with_vulnerabilities  # 无漏洞类别数量
        print(f"testset in lines of test set with vulnerabilities: {test_num_with_vulnerabilities}; lines without vulnerabilities:{test_num_without_vulnerabilities}")



        # np.random.shuffle(train_data)
        # np.random.shuffle(test_data)

        # 创建数据集和数据加载器
        if args.data_type == 'vectortype':
            train_dataset = VectorTypeDataset(train_data)
            test_dataset = VectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=VectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=VectorTypeCollate_fn)

            if args.pca == 'True':
                input_dim = 600
            else:
                input_dim = 1068
        elif args.data_type == 'vector':
            train_dataset = LineVectorDataset(train_data)
            test_dataset = LineVectorDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LineVectorCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=LineVectorCollate_fn)

            input_dim = 768
        elif args.data_type == 'type':
            train_dataset = LineTypeDataset(train_data)
            test_dataset = LineTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LineTypecollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=LineTypecollate_fn)

            input_dim = 300

        elif args.data_type == 'graph_vectortype':
            train_dataset = GraphVectorTypeDataset(train_data)
            test_dataset = GraphVectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=GraphVectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=GraphVectorTypeCollate_fn)

            if args.pca == 'True':
                input_dim = 600
            else:
                input_dim = 1068



        elif args.data_type == 'graph_vector' or args.data_type == 'graph_type':
            train_dataset = GraphVectorTypeDataset(train_data)
            test_dataset = GraphVectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=GraphVectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=GraphVectorTypeCollate_fn)

            input_dim = 768 if args.data_type == 'graph_vector' else 300
        #
        print(input_dim)



        # 定义模型
        if args.model == 'gat_lstm':
            model = GraphLSTMModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_lstm_contract':
            model = GraphLSTMContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)




        if args.mode == 'train':
            model.train(num_epochs=args.epochs, dataloader=train_dataloader, test_dataloader=test_dataloader)
            result = model.test(dataloader=test_dataloader)
            save_modle_result(result)

        else:

            state_dict = torch.load(f'./model/pth/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.pth')
            model.model.load_state_dict(state_dict)

            # count time
            with torch.no_grad():
                for _ in range(10):
                    _ = model.test(dataloader=test_dataloader)

            n_runs = 100
            torch.cuda.synchronize() 
            start = time.perf_counter()

            with torch.no_grad():
                for _ in range(n_runs):
                    _ = model.test(dataloader=test_dataloader)

            torch.cuda.synchronize()  
            end = time.perf_counter()

            avg_time_ms = (end - start) / n_runs * 1000
            print(f'Average inference time per batch: {avg_time_ms:.2f} ms')
