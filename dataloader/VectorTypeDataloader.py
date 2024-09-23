# 使用行的向量拼接类别作为输入

import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import fasttext
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English

ft = fasttext.load_model('cc.en.300.bin')
# fasttext.util.reduce_model(ft, 100)  # 降低维度至100
# ft.get_word_vector('hello')

def TyepEmbedding(typelist):
    type_vectors = []
    # 将类别词转换为向量
    for line_types in typelist:
        line_type_vectors = [ft.get_word_vector(type) for type in line_types]
        # 计算平均向量
        average_vector = sum(line_type_vectors) / len(line_type_vectors)
        type_vectors.append(average_vector)

    return type_vectors

class VectorTypeDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset


    def __getitem__(self, idx):
        contract, vectors, line_labels, flag_labels  = self.data[idx]
        return contract, torch.tensor(vectors, dtype=torch.float32), torch.tensor(line_labels, dtype=torch.long), torch.tensor(flag_labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

def VectorTypeCollate_fn(batch):
    # Find the max length in this batch
    max_length = max(len(vectors) for _, vectors, _, _ in batch)

    contracts = []
    flag_labels = []
    padded_vectors = []
    padded_labels = []
    masks = [] # 设置掩码区分实际数据和填充值

    for contract, vectors, labels,flag  in batch:
        # Padding vectors and labels to the max length
        pad_size = max_length - len(vectors)
        padded_vectors.append(torch.cat([vectors, torch.zeros((pad_size, vectors.size(1)))], dim=0))
        # padded_labels.append(torch.cat([labels, torch.zeros(pad_size, dtype=torch.long)], dim=0))
        padded_labels.append(torch.cat([labels, torch.full((pad_size,),-1, dtype=torch.long)], dim=0))

        # mask需要的是一个矩阵形式
        masks.append(torch.cat([torch.ones(len(vectors)), torch.zeros(pad_size)], dim=0))
        # mask_matrix = torch.ones((len(vectors), len(vectors)))
        # mask_matrix = torch.nn.functional.pad(mask_matrix, pad=(0, pad_size, 0, pad_size), value=0)
        # masks.append(mask_matrix)

        contracts.append(contract)
        flag_labels.append(flag)

    # Stack the padded vectors, labels and masks
    padded_vectors = torch.stack(padded_vectors)
    padded_labels = torch.stack(padded_labels)
    masks = torch.stack(masks)
    flag_labels= torch.stack(flag_labels)

    return contracts, padded_vectors, padded_labels, masks, flag_labels

# # Example usage:
# vectors_file = '../data/ge-sc-data/source_code/access_control/LineVectorList.json'
# types_file = '../data/ge-sc-data/source_code/access_control/typeList.json'
# labels_file = '../data/ge-sc-data/source_code/access_control/craft_preprocess_all.json'
#
#
# dataset = VectorTypeDataset(vectors_file, labels_file, types_file)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
#
# for batch_idx, (contracts, vectors, labels, masks, flags) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}")
#     print(f"Contracts: {contracts}")
#     print(f"Vectors: {vectors}")
#     print(f"Labels: {labels}")
#     print(f"Masks: {masks}")
#     print(f"Flags: {flags}")