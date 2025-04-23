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
# 用于取不同类别漏洞的组合
from itertools import combinations


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


# 总类型列表
all_types = [
        'access_control',
        'arithmetic',
        'denial_of_service',
        'front_running',
        'reentrancy',
        'time_manipulation',
        'unchecked_low_level_calls'
]




# 将测试模型的数据保存下来:name数据集名，model：模型类别
def save_modle_result(test_result, un_types_count):
    dic = {}
    key = ["tp", "fp", "tn", "fn", "accuracy", "precision", "recall",  "F1", "test_loss"]
    for index, value in enumerate(test_result):
        dic[key[index]] = value
    info_json = json.dumps(dic)

    type_path =  '_'.join(args.selecttype)

    # path = f"./result_2/all/{type_path}/sixlayer_{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    path = f"./result_2/all/{un_types_count}_sixlayer_{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    # 使用多个类别作为测试的时候开启下面的
    # path = f"./result_2/all/{type_path}_{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.json"
    with open(path, "a+") as f:
        # pickle.dump(data, my_file)
        f.write(info_json + "\n")




def read_contracts(typelist ):
    contracts_with_vulnerabilities = []
    contracts_without_vulnerabilities = []
    all_contracts = []

    for type in typelist:
        vectors_file = f'./data/ge-sc-data/source_code/{type}/LineVectorList.json'
        # 使用fasttext转换的行级语义特征
        # vectors_file = f'./data/ge-sc-data/source_code/{args.type}/LineVectorList_Fasttext.json'
        # 使用bert转换的行级语义特征
        # vectors_file = f'./data/ge-sc-data/source_code/{args.type}/LineVectorListBert.json'

        # labels_file = f'./data/ge-sc-data/source_code/{args.type}/craft_preprocess_all.json'
        labels_file = f'./data/ge-sc-data/source_code/{type}/preprocess_all.json'

        types_file = f'./data/ge-sc-data/source_code/{type}/typeList.json'

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
                contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvectortype_pca(type, vectors_file, labels_file, types_file)
            elif args.pca == 'add':
                contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvectortype_pca_add(type,
                    vectors_file, labels_file, types_file)
            else:
                contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvectortype(type, vectors_file, labels_file,
                                                                                                              types_file)
        elif args.data_type == 'graph_vector':
                contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphvector(type, vectors_file, labels_file)

        elif args.data_type == 'graph_type':
                contracts_with_vulnerabilities, contracts_without_vulnerabilities = Split_dataset_graphtype(type, types_file, labels_file)

        all_contracts += contracts_with_vulnerabilities + contracts_without_vulnerabilities

    return all_contracts

# 获取漏洞类型的所有不重复组合
def get_vulnerability_combinations(n: int) -> list[list[str]]:
    """
    获取漏洞类型的所有不重复组合

    :param types: 漏洞类型列表
    :param n: 组合中包含的类型个数
    :return: 所有不重复的组合列表，每个组合是一个列表
    """
    if n < 1 or n > len(all_types):
        raise ValueError(f"n should be between 1 and {len(all_types)}")

    return [list(combo) for combo in combinations(all_types, n)]


def train_all(test_types, train_times, un_types_count):
    # 取差集（排除选中的）这样做会导致无序
    # remaining_types = list(set(all_types) - set(test_types))
    # 这样做会是有序的
    remaining_types = [t for t in all_types if t not in test_types]

    train_data = read_contracts(remaining_types)
    test_data = read_contracts(test_types)

    for time in range(train_times):

        # 输出配置
        print(f'model:{args.model};         data_type:{args.data_type};      '
              f'vul_type:{args.selecttype};        epochs:{args.epochs};          model_dim:{args.model_dim}  mode:{args.pca}')
        print(f"train_contracts count:{len(train_data)}; test_contracts count:{len(test_data)}")
        class_weight = [1.0, 1.0]
        print(f"the {time} times to train the {test_types}")
        # 打乱数据
        # np.random.shuffle(train_data)
        # np.random.shuffle(test_data)

        # 创建数据集和数据加载器
        if args.mode == 'test':
            if args.data_type == 'vectortype':
                test_dataset = VectorTypeDataset(test_data)
                test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=VectorTypeCollate_fn)
            else:
                test_dataset = GraphVectorTypeDataset(test_data)
                test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                             collate_fn=GraphVectorTypeCollate_fn)

            if args.pca == 'True':
                input_dim = 600
            else:
                # 使用fastext时vector为300
                input_dim = 1068
                # input_dim = 600
        elif args.data_type == 'vectortype':
            train_dataset = VectorTypeDataset(train_data)
            test_dataset = VectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=VectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=VectorTypeCollate_fn)

            if args.pca == 'True':
                input_dim = 600
            else:
                # 使用fastext时vector为300
                input_dim = 1068
                # input_dim = 600
        elif args.data_type == 'vector':
            train_dataset = LineVectorDataset(train_data)
            test_dataset = LineVectorDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=LineVectorCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=LineVectorCollate_fn)

            # 使用fastext时vector为300
            input_dim = 768
            # input_dim = 300
        elif args.data_type == 'type':
            train_dataset = LineTypeDataset(train_data)
            test_dataset = LineTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=LineTypecollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=LineTypecollate_fn)

            input_dim = 300

        # 只是用gat进行合约级别漏洞检查。区别在于合约节点的维度是用vectortype还是vector
        elif args.data_type == 'graph_vectortype':
            train_dataset = GraphVectorTypeDataset(train_data)
            test_dataset = GraphVectorTypeDataset(test_data)

            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                          collate_fn=GraphVectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                         collate_fn=GraphVectorTypeCollate_fn)

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

            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                          collate_fn=GraphVectorTypeCollate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                         collate_fn=GraphVectorTypeCollate_fn)

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
            model = SingleTransformerModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'transformerencoder':
            model = TransformerEncoderModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        # 训练合约级别分类
        elif args.model == 'transformer_contract':
            model = SingleTransformerContractModel(input_dim=input_dim, model_dim=args.model_dim,
                                                   class_weight=class_weight)
        elif args.model == 'transformerencoder_contract':
            model = TransformerEncoderContractModel(input_dim=input_dim, model_dim=args.model_dim,
                                                    class_weight=class_weight)
        elif args.model == 'gat_contract':
            model = GATContractModel(input_dim=input_dim, model_dim=args.model_dim, class_weight=class_weight)
        elif args.model == 'gat_transformerencoder':
            # model = GraphTransformerEncoderModel(input_dim=input_dim, model_dim=args.model_dim  )
            model = GraphTransformerEncoderModelV2(input_dim=input_dim, model_dim=args.model_dim,
                                                   class_weight=class_weight)
        elif args.model == 'gat_transformerencoder_contract':
            model = GraphTransformerEncoderContractModel(input_dim=input_dim, model_dim=args.model_dim,
                                                         class_weight=class_weight)
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
            save_modle_result(result,un_types_count)

        else:
            # date_time = '06-27-09-58' # transformer
            # date_time = '07-01-08-37'   # transformer_contract
            # model.model = torch.load(f'./model/pth/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{date_time}.pth')

            # 读取GAT模型
            # state_dict = torch.load(f'./model/pth/{args.type}/'
            #                         f'{args.data_type}_{args.model}_{args.model_dim}'
            #                         f'_{args.epochs}_{args.pca}.pth')
            # model.model.load_state_dict(state_dict)

            result = model.test(dataloader=test_dataloader)
            save_modle_result(result,'test')


if __name__ == '__main__':

    # 指定某一个类别作为测试集的时候
    # test_types = args.selecttype
    # train_all(test_types, 2)
    for i in range(7,8):
        un_types_count = i
        # 使用n个类别作为测试集
        test_types = get_vulnerability_combinations(un_types_count)

        for _,types in enumerate(test_types):
            train_all(types, 1,un_types_count)
