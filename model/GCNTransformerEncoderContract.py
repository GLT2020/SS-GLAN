
import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import random

from parser1 import parameter_parser
args = parameter_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

LR = 0.00001


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     dgl.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(42)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Flatten the inputs and targets
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)

        # Mask to ignore specific index
        ignore_mask = targets != self.ignore_index

        # Filter out the ignored targets
        targets = targets[ignore_mask]
        inputs = inputs[ignore_mask]

        # Compute the standard cross-entropy loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss



class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, graph_feat):
        # src: [seq_len, batch_size, d_model]
        # graph_feat: [num_nodes, batch_size, d_model]
        q = src
        k = v = graph_feat
        # attn_output, _ = self.cross_attn(q, k, v, key_padding_mask=graph_mask)
        attn_output, _ = self.cross_attn(q, k, v)

        # # TODO:添加跳连
        src = src + self.dropout(attn_output)
        src = self.norm(src)
        return src
        # return attn_output

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)


    def forward(self, g, inputs):
        gnn_x = inputs
        gnn_x = self.conv1(g, gnn_x)

        gnn_x = self.conv2(g, gnn_x)

        gnn_x = self.conv3(g, gnn_x)

        g.ndata['h'] = gnn_x

        # 拆解回[batch, num_nodes, node_features]
        unbatched_graphs = dgl.unbatch(g)
        node_features = [i.ndata['h'] for i in unbatched_graphs]

        # Pad the node features to ensure the shape [batch_size, max_num_nodes, nodes_features]
        max_num_nodes = max(f.shape[0] for f in node_features)
        padded_features = [torch.cat([f, torch.zeros(max_num_nodes - f.shape[0], f.shape[1]).to(device)], dim=0) for f in
                           node_features]
        node_features = torch.stack(padded_features)

        # # 使用平均值代表一个图
        # unbatched_graphs = dgl.unbatch(g)
        # node_features = [dgl.mean_nodes(graph, 'h') for graph in unbatched_graphs]
        # # 调整形状 [batch_size, 1, nodes_features]
        # node_features = torch.stack(node_features)

        return  node_features





class CustomGraphTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(CustomGraphTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)

        self.gat_layer = GCNModel(input_dim=input_dim, hidden_dim=model_dim).to(device)

        # # TODO:使用pretrain GAT
        # state_dict = torch.load(f'./model/pth/{args.type}/graph_vectortype_gat_contract_{args.model_dim}_{args.epochs}_{args.pca}.pth')
        # state_dict = {k: v for k, v in state_dict.items() if 'class' not in k}  # 移除与分类层相关的部分
        # self.gat_layer.load_state_dict(state_dict)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # TODO:使用pretrain Transformer Encoder
        # self.load_pretrained_transformer(f'./model/pth/{args.type}/vectortype_transformerencoder_{args.model_dim}_{args.epochs}_{args.pca}.pth')

        self.cross_attn_layer = CrossAttentionLayer(model_dim, num_heads)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def load_pretrained_transformer(self, path):
        state_dict = torch.load(path, map_location=device)
        model_dict = self.transformer_encoder.state_dict()
        # Remove prefix 'transformer_encoder.' from pretrained model keys
        pretrained_dict = {k[len('transformer_encoder.'):]: v for k, v in state_dict.items() if
                           k.startswith('transformer_encoder.')}
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.transformer_encoder.load_state_dict(model_dict)


    def forward(self, src, src_mask, graph, graph_feat):
        # Embedding
        src = self.embedding(src)  # [batch_size, seq_len, model_dim]

        # GAT Layer
        graph_feat = self.gat_layer(graph, graph_feat)
        src = self.cross_attn_layer(src, graph_feat)

        # Encoder with Cross Attention Layer
        for layer in self.transformer_encoder.layers:
            src = layer(src, src_key_padding_mask=~src_mask.bool())
            # src = self.cross_attn_layer(src, graph_feat)

        src = self.cross_attn_layer(src, graph_feat)

        # Apply global average pooling
        src = src.permute(0, 2, 1)  # Change shape to (batch_size, model_dim, seq_len)
        pooled_output = self.pooling(src).squeeze(-1)  # Change shape to (batch_size, model_dim)

        # Final classification
        output = self.classifier(pooled_output)
        return output


class GCNTransformerEncoderContractModel():
    def __init__(self, input_dim, model_dim, class_weight, num_classes=2, num_heads=8, num_layer=6):
        self.model = CustomGraphTransformer(input_dim=input_dim, model_dim=model_dim, num_classes=num_classes,
                                       num_heads=num_heads, num_layers=num_layer).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(device), ignore_index=-1)

        # 使用带有ignore_index的Focal Loss
        # self.criterion = FocalLoss(alpha=1, gamma=2, reduction='mean', ignore_index=-1)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=150, eta_min=1e-6)


    def train(self, num_epochs, dataloader, test_dataloader):
        self.model.train()
        date = datetime.today()
        date_time = date.strftime('%m-%d-%H-%M')

        for epoch in range(num_epochs):

            # # 手动调整学习率
            # if epoch == 250:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] *= 0.01  # 将学习率减少

            epoch_loss = 0
            for batch_idx, (contracts, vectors, labels, masks, flag_labels, graphs) in enumerate(dataloader):
                np_vectors = vectors.numpy()
                vectors = vectors.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                flag_labels = flag_labels.to(device)
                graphs = graphs.to(device)
                graph_feat = graphs.ndata['feat'].to(device)

                self.optimizer.zero_grad()
                output = self.model(vectors, masks, graphs, graph_feat)

                loss = self.criterion(output, flag_labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
            if test_dataloader and epoch % 10 ==0 :
                # current_lr = self.scheduler.get_last_lr()[0]
                # print(current_lr)
                val_loss = self.test(test_dataloader)
                # self.scheduler.step()

        # torch.save(self.model, f'./model/pth/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{date_time}.pth')

    def test(self, dataloader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for batch_idx, (contracts, vectors, labels, masks, flag_labels, graphs) in enumerate(dataloader):
                vectors = vectors.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                flag_labels = flag_labels.to(device)
                graphs = graphs.to(device)
                graph_feat = graphs.ndata['feat'].to(device)

                self.optimizer.zero_grad()
                output = self.model(vectors, masks, graphs, graph_feat)

                loss = self.criterion(output, flag_labels)

                test_loss += loss.item()
                # Get the index of the max log-probability
                pred = output.detach().argmax(dim=1, keepdim=True)
                correct += pred.eq(flag_labels.view_as(pred)).sum().item()
                total += flag_labels.size(0)

                pred_np = np.array(pred.cpu())
                labels_np = np.array(flag_labels.cpu())

                # Calculate TP, FP, FN
                true_positives += ((pred == 1) & (flag_labels.view_as(pred) == 1)).sum().item()
                true_negatives += ((pred == 0) & (flag_labels.view_as(pred) == 0)).sum().item()
                false_positives += ((pred == 1) & (flag_labels.view_as(pred) == 0)).sum().item()
                false_negatives += ((pred == 0) & (flag_labels.view_as(pred) == 1)).sum().item()

        test_loss /= len(dataloader)
        accuracy = correct / total
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # print(f"total:{total}; correct:{correct}; tp:{true_positives}; tn:{true_negatives} ;fp:{false_positives}; fn:{false_negatives}")
        # print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return (true_positives, false_positives, true_negatives, false_negatives, accuracy, precision, recall, f1_score, test_loss)
