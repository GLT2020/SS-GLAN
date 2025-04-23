import dgl.nn as dglnn
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from datetime import datetime


from parser1 import parameter_parser
args = parameter_parser()
use_cuda = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

LR = 0.00001

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GCNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)

        # output
        self.class1 = nn.Linear(hidden_dim, 512)
        self.class2 = nn.Linear(512, 256)
        self.class3 = nn.Linear(256, out_dim)

    def forward(self, g, inputs):
        gnn_x = inputs
        gnn_x = self.conv1(g, gnn_x)

        gnn_x = self.conv2(g, gnn_x)

        gnn_x = self.conv3(g, gnn_x)

        g.ndata['h'] = gnn_x
        # 以平均值来代表图
        gnn_output = dgl.mean_nodes(g, 'h')

        # classify
        x = self.class1(gnn_output)
        x = self.relu(x)
        x = self.class2(x)
        x = self.relu(x)
        x = self.class3(x)
        x = self.relu(x)
        return x


class GCNContractModel():
    def __init__(self, input_dim, model_dim, class_weight,num_classes=2):
        self.model = GCNModel(input_dim=input_dim, hidden_dim=model_dim, out_dim=num_classes).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(device), ignore_index=-1)
        # self.criterion = nn.CrossEntropyLoss()

        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [100,150,200,250], gamma=0.1)  # dynamic adjustment lr


    def train(self, num_epochs, dataloader, test_dataloader):
        self.model.train()
        date = datetime.today()
        date_time = date.strftime('%m-%d-%H-%M')

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (_, _, _, _, flag_labels, graphs) in enumerate(dataloader):

                flag_labels = flag_labels.to(device)
                graphs = graphs.to(device)
                graph_feature = graphs.ndata['feat'].to(device)

                self.optimizer.zero_grad()
                output = self.model(graphs, graph_feature)


                loss = self.criterion(output, flag_labels)
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()  # 调整lr

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

            if test_dataloader and epoch % 10 == 0:
                val_loss = self.test(test_dataloader)

        # torch.save(self.model, f'./model/pth/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{date_time}.pth')
        # torch.save(self.model.state_dict(), f'./model/pth/{args.type}/{args.data_type}_{args.model}_{args.model_dim}_{args.epochs}_{args.pca}.pth')

    def test(self, dataloader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for batch_idx, (_, _, _, _, flag_labels, graphs) in enumerate(dataloader):
            flag_labels = flag_labels.to(device)
            graphs = graphs.to(device)
            graph_feature = graphs.ndata['feat'].to(device)

            self.optimizer.zero_grad()
            output = self.model(graphs, graph_feature)

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

        # print(
        #     f"total:{total}; correct:{correct}; tp:{true_positives}; tn:{true_negatives} ;fp:{false_positives}; fn:{false_negatives}")
        # print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        return (true_positives, false_positives, true_negatives, false_negatives, accuracy, precision, recall, f1_score, test_loss)