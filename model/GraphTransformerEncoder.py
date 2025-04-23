import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
import numpy as np
import os

from parser1 import parameter_parser
args = parameter_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

# LR = 0.00001
LR = 0.00001

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, src, graph_feat):
        # src: [seq_len, batch_size, d_model]
        # graph_feat: [num_nodes, batch_size, d_model]
        q = src
        k = v = graph_feat
        # attn_output, _ = self.cross_attn(q, k, v, key_padding_mask=graph_mask)
        attn_output, _ = self.cross_attn(q, k, v)
        return attn_output

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = dglnn.GATConv(input_dim, hidden_dim, num_heads)
        self.conv2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.conv3 = dglnn.GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
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

        # Pad the node features to ensure the shape [batch_size, max_num_nodes, nodes_features]
        max_num_nodes = max(f.shape[0] for f in node_features)
        padded_features = [torch.cat([f, torch.zeros(max_num_nodes - f.shape[0], f.shape[1]).to(device)], dim=0) for f in
                           node_features]
        node_features = torch.stack(padded_features)

        return  node_features


class CustomGraphTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(CustomGraphTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.gat_layer =  GATModel(input_dim=input_dim, hidden_dim=model_dim, num_heads=num_heads).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.cross_attn_layer = CrossAttentionLayer(model_dim, num_heads)
        self.classifier = nn.Linear(model_dim, num_classes)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, src_mask, graph, graph_feat):
        # Embedding
        src = self.embedding(src)  # [batch_size, seq_len, model_dim]
        # src = src.transpose(0, 1)  # [seq_len, batch_size, model_dim]

        # GAT Layer
        graph_feat = self.gat_layer(graph, graph_feat)

        # Encoder with Cross Attention Layer
        for layer in self.encoder.layers:
            src = layer(src, src_key_padding_mask=~src_mask.bool())
            src = self.cross_attn_layer(src, graph_feat)

        # Final classification
        # src = src.transpose(0, 1)  # [batch_size, seq_len, model_dim]
        output = self.classifier(src)
        return output


class GraphTransformerEncoderModel():
    def __init__(self, input_dim, model_dim, class_weight, num_classes=2, num_heads=8, num_layer=6):
        self.model = CustomGraphTransformer(input_dim=input_dim, model_dim=model_dim, num_classes=num_classes,
                                       num_heads=num_heads, num_layers=num_layer).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(device), ignore_index=-1)

    def train(self, num_epochs, dataloader):
        self.model.train()
        date = datetime.today()
        date_time = date.strftime('%m-%d-%H-%M')

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (contracts, vectors, labels, masks, flag_labels, graphs) in enumerate(dataloader):
                vectors = vectors.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                flag_labels = flag_labels.to(device)
                graphs = graphs.to(device)
                graph_feat = graphs.ndata['feat'].to(device)

                self.optimizer.zero_grad()
                output = self.model(vectors, masks, graphs, graph_feat)

                # Flatten the output and labels for loss calculation
                output = output.view(-1, output.size(-1))
                labels = labels.view(-1)

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

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

        for batch_idx, (contracts, vectors, labels, masks, flag_labels, graphs) in enumerate(dataloader):
            vectors = vectors.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            flag_labels = flag_labels.to(device)
            graphs = graphs.to(device)
            graph_feat = graphs.ndata['feat'].to(device)

            self.optimizer.zero_grad()
            output = self.model(vectors, masks, graphs, graph_feat)

            # Flatten the output and labels for loss calculation
            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            loss = self.criterion(output, labels)

            # Remove padding elements
            non_padding_mask = labels != -1
            output = output[non_padding_mask]
            labels = labels[non_padding_mask]

            test_loss += loss.item()
            # Get the index of the max log-probability
            pred = output.detach().argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)

            pred_np = np.array(pred.cpu())
            labels_np = np.array(labels.cpu())

            # Calculate TP, FP, FN
            true_positives += ((pred == 1) & (labels.view_as(pred) == 1)).sum().item()
            true_negatives += ((pred == 0) & (labels.view_as(pred) == 0)).sum().item()
            false_positives += ((pred == 1) & (labels.view_as(pred) == 0)).sum().item()
            false_negatives += ((pred == 0) & (labels.view_as(pred) == 1)).sum().item()

        test_loss /= len(dataloader)
        accuracy = correct / total
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"total:{total}; correct:{correct}; tp:{true_positives}; tn:{true_negatives} ;fp:{false_positives}; fn:{false_negatives}")
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return (true_positives, false_positives, true_negatives, false_negatives, accuracy, precision, recall, f1_score, test_loss)
