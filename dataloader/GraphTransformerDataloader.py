import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import dgl

class GraphTransformerDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __getitem__(self, idx):
        contract, vectors, line_labels, flag_labels, dgl_graph = self.data[idx]
        return contract, \
               torch.tensor(vectors, dtype=torch.float32), \
               torch.tensor(line_labels, dtype=torch.long), \
               torch.tensor(flag_labels, dtype=torch.long), \
               dgl_graph

    def __len__(self):
        return len(self.data)

# def pad_graph_feats(graph_feats):
#     max_num_nodes = max([feat.size(0) for feat in graph_feats])
#     padded_graph_feats = pad_sequence([F.pad(feat, (0, 0, 0, max_num_nodes - feat.size(0))) for feat in graph_feats], batch_first=True)
#     return padded_graph_feats

def GraphTransformerCollate_fn(batch):
    max_length = max(len(vectors) for _, vectors, _, _, _ in batch)

    contracts = []
    flag_labels = []
    padded_vectors = []
    padded_labels = []
    masks = []  # 设置掩码区分实际数据和填充值
    graphs = []
    graph_feats = []
    # graph_masks = []

    for contract, vectors, labels, flag, dgl_graph in batch:
        pad_size = max_length - len(vectors)
        padded_vectors.append(torch.cat([vectors, torch.zeros((pad_size, vectors.size(1)))], dim=0))
        padded_labels.append(torch.cat([labels, torch.full((pad_size,), -1, dtype=torch.long)], dim=0))
        masks.append(torch.cat([torch.ones(len(vectors)), torch.zeros(pad_size)], dim=0))

        contracts.append(contract)
        flag_labels.append(flag)
        graphs.append(dgl_graph)
        graph_feats.append(dgl_graph.ndata['feat'])
        # graph_masks.append(torch.ones(dgl_graph.num_nodes(), dtype=torch.bool))

    padded_vectors = torch.stack(padded_vectors)
    padded_labels = torch.stack(padded_labels)
    masks = torch.stack(masks)
    flag_labels = torch.stack(flag_labels)
    batched_graph = dgl.batch(graphs)

    # padded_graph_feats = pad_graph_feats(graph_feats)
    # graph_masks = pad_graph_feats(graph_masks)

    # return contracts, padded_vectors, padded_labels, masks, flag_labels, batched_graph, padded_graph_feats, graph_masks
    return contracts, padded_vectors, padded_labels, masks, flag_labels, batched_graph