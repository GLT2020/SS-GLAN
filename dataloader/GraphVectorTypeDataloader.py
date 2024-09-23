import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl

import fasttext
import fasttext.util


class GraphVectorTypeDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset


    def __getitem__(self, idx):
        contract, vectors, line_labels, flag_labels, dgl_graph = self.data[idx]
        return contract, torch.tensor(vectors, dtype=torch.float32), \
               torch.tensor(line_labels, dtype=torch.long), \
               torch.tensor(flag_labels, dtype=torch.long), dgl_graph

    def __len__(self):
        return len(self.data)

def GraphVectorTypeCollate_fn(batch):
    # Find the max length in this batch
    max_length = max(len(vectors) for _, vectors, _, _,_ in batch)

    contracts = []
    flag_labels = []
    padded_vectors = []
    padded_labels = []
    masks = [] # 设置掩码区分实际数据和填充值
    graphs = []


    for contract, vectors, labels, flag, dgl_graph  in batch:
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
        graphs.append(dgl_graph)

    # Stack the padded vectors, labels and masks
    padded_vectors = torch.stack(padded_vectors)
    padded_labels = torch.stack(padded_labels)
    masks = torch.stack(masks)
    flag_labels= torch.stack(flag_labels)
    batched_graph = dgl.batch(graphs)

    return contracts, padded_vectors, padded_labels, masks, flag_labels, batched_graph

