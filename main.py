import json
import torch
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random

from dataloader.VectorTypeDataloader import VectorTypeDataset, VectorTypeCollate_fn
from dataloader.LineVectorDataloader import LineVectorDataset, LineVectorCollate_fn
from dataloader.LineTypeDataloader import LineTypeDataset, LineTypecollate_fn
from dataloader.GraphVectorTypeDataloader import GraphVectorTypeDataset, GraphVectorTypeCollate_fn
from dataloader.GraphTransformerDataloader import GraphTransformerDataset, GraphTransformerCollate_fn

from model.SingleTransformer import SingleTransformerModel
from model.TransformerEncoder import TransformerEncoderModel
from model.SingleTransformerContract import SingleTransformerContractModel
from model.TransformerEncoderContract import TransformerEncoderContractModel
from model.GATContract import GATContractModel
from model.GraphTransformerEncoder import GraphTransformerEncoderModel
from model.GraphTransformerEncoderV2 import GraphTransformerEncoderModelV2
from model.GCNTransformerEncoderV2 import GCNTransformerEncoderModel
from model.GraphTransformerEncoderContract import GraphTransformerEncoderContractModel
from model.GCNTransformerEncoderContract import GCNTransformerEncoderContractModel
from model.GCNContract import GCNContractModel

from model.GraphLSTM import GraphLSTMModel
from model.LSTM import LSTMModel
from model.GraphLSTMContract import GraphLSTMContractModel
from model.LSTMContract import LSTMContractModel

from model.GraphGRU import GraphGRUModel
from model.GRU import GRUModel
from model.GraphGRUContract import GraphGRUContractModel
from model.GRUContract import GRUContractModel


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

    # path = f"./result_2/{args.type}/fasttext_{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    path = f"./result_2/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    # path = f"./result_2/{args.type}/minlayer_{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    with open(path, "a+") as f:
        # pickle.dump(data, my_file)
        f.write(info_json + "\n")



if __name__ == '__main__':
    vectors_file = f'./data/ge-sc-data/source_code/{args.type}/LineVectorList.json'
    # 使用fasttext转换的行级语义特征
    # vectors_file = f'./data/ge-sc-data/source_code/{args.type}/LineVectorList_Fasttext.json'
    # 使用bert转换的行级语义特征
    # vectors_file = f'./data/ge-sc-data/source_code/{args.type}/LineVectorListBert.json'

    # labels_file = f'./data/ge-sc-data/source_code/{args.type}/craft_preprocess_all.json'
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
        elif args.pca == 'add':
            contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvectortype_pca_add(args.type,
                vectors_file, labels_file, types_file)
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


    for time in range(7):
        print(f"the {time} times to train the {args.type}")
        # 输出配置
        print(f'model:{args.model};         data_type:{args.data_type};      '
              f'vul_type:{args.type};        epochs:{args.epochs};          model_dim:{args.model_dim}  mode:{args.pca}')
        print(f'all dataset in num_without_vulnerabilities:{num_without_vulnerabilities};       num_with_vulnerabilities：{num_with_vulnerabilities} ')
        # TODO: 打开权重
        # class_weight = [num_without_vulnerabilities/total_samples*0.1, num_with_vulnerabilities/total_samples*1.1]
        # class_weight = [num_without_vulnerabilities/total_samples , num_with_vulnerabilities/total_samples ]
        class_weight = [1.0, 1.0]

        print(f'class weight:{class_weight}')

        # 切分数据集，按照7:3比例
        train_with, test_with = train_test_split(contracts_with_vulnerabilities, test_size=0.3)
        train_without, test_without = train_test_split(contracts_without_vulnerabilities, test_size=0.3)
        # train_with, test_with = train_test_split(contracts_with_vulnerabilities, test_size=0.3, random_state=42)
        # train_without, test_without = train_test_split(contracts_without_vulnerabilities, test_size=0.3, random_state=42)

        # 合并训练集和测试集
        train_data = train_with + train_without
        test_data = test_with + test_without

        # # 将固定切分的数据集合约名字输出，以便其他实验使用
        # train_data_name = [(sublabel[0], sublabel[3]) for sublabel in train_data ]
        # test_data_name = [(sublabel[0], sublabel[3]) for sublabel in test_data]
        # with open(f'./ge_label/{args.type}/trainset_name.csv', 'w', encoding='utf-8') as f:
        #     for name, label in train_data_name:
        #         f.write(f"{str(name)},{str(label)}\n")  # 每个文件名后添加换行符
        #     print("successful store trainset name")
        # with open(f'./ge_label/{args.type}/testset_name.csv', 'w', encoding='utf-8') as f:
        #     for name, label in test_data_name:
        #         f.write(f"{str(name)},{str(label)}\n")  # 每个文件名后添加换行符
        #     print("successful store testset name")

        # 计算测试集有漏洞行数和无漏洞函数
        test_labels = [item for sublabel in test_data for item in sublabel[2]]
        test_total_samples = len(test_labels)
        test_num_with_vulnerabilities = sum(1 for labels in test_labels if labels)  # 漏洞类别数量
        test_num_without_vulnerabilities = test_total_samples - test_num_with_vulnerabilities  # 无漏洞类别数量
        print(f"testset in lines of test set with vulnerabilities: {test_num_with_vulnerabilities}; lines without vulnerabilities:{test_num_without_vulnerabilities}")



        # 打乱数据
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
                #使用fastext时vector为300
                input_dim = 1068
                # input_dim = 600
        elif args.data_type == 'vector':
            train_dataset = LineVectorDataset(train_data)
            test_dataset = LineVectorDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LineVectorCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=LineVectorCollate_fn)

            # 使用fastext时vector为300
            input_dim = 768
            # input_dim = 300
        elif args.data_type == 'type':
            train_dataset = LineTypeDataset(train_data)
            test_dataset = LineTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=LineTypecollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=LineTypecollate_fn)

            input_dim = 300

        # 只是用gat进行合约级别漏洞检查。区别在于合约节点的维度是用vectortype还是vector
        elif args.data_type == 'graph_vectortype':
            train_dataset = GraphVectorTypeDataset(train_data)
            test_dataset = GraphVectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=GraphVectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=GraphVectorTypeCollate_fn)

            if args.pca == 'True':
                input_dim = 600
            elif args.pca == 'add':
                input_dim = 300
            else:
                # 使用fastext时vector为300
                input_dim = 1068
                # input_dim = 600


        elif args.data_type == 'graph_vector' or args.data_type == 'graph_type':
            train_dataset = GraphVectorTypeDataset(train_data)
            test_dataset = GraphVectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=GraphVectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=GraphVectorTypeCollate_fn)

            # 使用graph_vector就是768， 使用graph_type就是300
            input_dim = 768 if args.data_type == 'graph_vector' else 300
            # 使用fastext时vector为300
            # input_dim = 300 if args.data_type == 'graph_vector' else 300

        print(input_dim)
        # graph结合transformer
        # elif args.data_type == 'graph_vectortype_transformer' or args.data_type == 'graph_vector_transformer':
        #     train_dataset = GraphTransformerDataset(train_data)
        #     test_dataset = GraphTransformerDataset(test_data)
        #
        #     train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=GraphTransformerCollate_fn)
        #     test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=GraphTransformerCollate_fn)
        #
        #     input_dim = 600 if args.pca == True else 1068
        #     if args.data_type == 'graph_vector_transformer':
        #         input_dim = 768



        # 定义模型
        if args.model == 'transformer':
            model = SingleTransformerModel(input_dim= input_dim, model_dim= args.model_dim, class_weight=class_weight)
        elif  args.model == 'transformerencoder':
            model = TransformerEncoderModel(input_dim= input_dim, model_dim= args.model_dim, class_weight=class_weight)
        # 训练合约级别分类
        elif args.model == 'transformer_contract':
            model = SingleTransformerContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'transformerencoder_contract':
            model = TransformerEncoderContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_contract':
            model = GATContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_transformerencoder':
            # model = GraphTransformerEncoderModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
            model = GraphTransformerEncoderModelV2(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_transformerencoder_contract':
            model = GraphTransformerEncoderContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gcn_transformerencoder':
            model = GCNTransformerEncoderModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gcn_transformerencoder_contract':
            model = GCNTransformerEncoderContractModel(input_dim=input_dim, model_dim=args.model_dim,
                                                         class_weight=class_weight)
        elif args.model == 'gcn_contract':
            model = GCNContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)

        elif args.model == 'gat_lstm':
            model = GraphLSTMModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'lstm':
            model = LSTMModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_lstm_contract':
            model = GraphLSTMContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'lstm_contract':
            model = LSTMContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)

        elif args.model == 'gat_gru':
            model = GraphGRUModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gru':
            model = GRUModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_gru_contract':
            model = GraphGRUContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gru_contract':
            model = GRUContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)




        if args.mode == 'train':
            # model.train(num_epochs=args.epochs, dataloader=train_dataloader)
            model.train(num_epochs=args.epochs, dataloader=train_dataloader, test_dataloader=test_dataloader)
            result = model.test(dataloader=test_dataloader)
            save_modle_result(result)

        else:
            # date_time = '06-27-09-58' # transformer
            # date_time = '07-01-08-37'   # transformer_contract
            # model.model = torch.load(f'./model/pth/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{date_time}.pth')

            # 读取GAT模型
            state_dict = torch.load(f'./model/pth/{args.type}/'
                                     f'{args.data_type}_{args.model}_{args.model_dim}'
                                     f'_{args.epochs}_{args.pca}.pth')
            model.model.load_state_dict(state_dict)

            result =  model.test(dataloader=test_dataloader)
            # save_modle_result(result)
