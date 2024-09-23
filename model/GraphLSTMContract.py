import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
import os

from parser1 import parameter_parser
args = parameter_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

LR = 0.00001


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
        q = src
        k = v = graph_feat
        attn_output, _ = self.cross_attn(q, k, v)
        src = src + self.dropout(attn_output)
        src = self.norm(src)
        return src

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = dglnn.GATConv(input_dim, hidden_dim, num_heads)
        self.conv2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.conv3 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, inputs):
        gnn_x = inputs
        gnn_x = self.conv1(g, gnn_x).flatten(1)
        gnn_x = self.conv2(g, gnn_x).flatten(1)
        gnn_x = self.conv3(g, gnn_x).mean(1)
        g.ndata['h'] = gnn_x

        # 拆解回[batch, num_nodes, node_features]
        unbatched_graphs = dgl.unbatch(g)
        node_features = [i.ndata['h'] for i in unbatched_graphs]

        # # Pad the node features to ensure the shape [batch_size, max_num_nodes, nodes_features]
        # max_num_nodes = max(f.shape[0] for f in node_features)
        # padded_features = [torch.cat([f, torch.zeros(max_num_nodes - f.shape[0], f.shape[1]).to(device)], dim=0) for f in
        #                    node_features]
        # node_features = torch.stack(padded_features)

        # 使用平均值代表一个图
        unbatched_graphs = dgl.unbatch(g)
        node_features = [dgl.mean_nodes(graph, 'h') for graph in unbatched_graphs]
        node_features = torch.stack(node_features)

        return node_features

class CustomGraphLSTM(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(CustomGraphLSTM, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.gat_layer = GATModel(input_dim=input_dim, hidden_dim=model_dim, num_heads=num_heads).to(device)
        # self.lstm = nn.LSTM(input_size=model_dim, hidden_size=model_dim, num_layers=num_layers,
        #                     batch_first=True, bidirectional=True, dropout=dropout)
        # self.cross_attn_layer = CrossAttentionLayer(model_dim * 2, num_heads)
        # self.classifier = nn.Linear(model_dim * 2, num_classes)

        self.lstm = nn.LSTM(input_size=model_dim, hidden_size=model_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout)
        self.cross_attn_layer = CrossAttentionLayer(model_dim, num_heads)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, src, src_mask, graph, graph_feat):
        src = self.embedding(src)
        graph_feat = self.gat_layer(graph, graph_feat)
        lstm_out, _ = self.lstm(src)
        src = self.cross_attn_layer(lstm_out, graph_feat)

        # Apply global average pooling
        src = src.permute(0, 2, 1)  # Change shape to (batch_size, model_dim, seq_len)
        pooled_output = self.pooling(src).squeeze(-1)  # Change shape to (batch_size, model_dim)

        # Final classification
        output = self.classifier(pooled_output)

        return output

class GraphLSTMContractModel():
    def __init__(self, input_dim, model_dim, class_weight, num_classes=2, num_heads=8, num_layer=6):
        self.model = CustomGraphLSTM(input_dim=input_dim, model_dim=model_dim, num_classes=num_classes,
                                       num_heads=num_heads, num_layers=num_layer).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(device), ignore_index=-1)
        self.criterion = FocalLoss(alpha=0.6, gamma=2, reduction='mean', ignore_index=-1)

    def train(self, num_epochs, dataloader, test_dataloader):
        self.model.train()
        date = datetime.today()
        date_time = date.strftime('%m-%d-%H-%M')

        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.train()
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
            if test_dataloader and epoch % 10 ==0:
                val_loss = self.test(test_dataloader)


    def test(self, dataloader):
        self.model.eval()
        # torch.backends.cudnn.enabled = False  # 禁用 cuDNN RNN 加速
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

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return (true_positives, false_positives, true_negatives, false_negatives, accuracy, precision, recall, f1_score, test_loss)
