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



class CustomGRUContract(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(CustomGRUContract, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)

        self.gru = nn.GRU(input_size=model_dim, hidden_size=model_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=False, dropout=dropout)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, src, src_mask):
        src = self.embedding(src)
        gru_out, _ = self.gru(src)

        # Apply global average pooling
        memory = gru_out.permute(0, 2, 1)  # Change shape to (batch_size, model_dim, seq_len)
        pooled_output = self.pooling(memory).squeeze(-1)  # Change shape to (batch_size, model_dim)

        # Final classification
        output = self.classifier(pooled_output)

        return output

class GRUContractModel():
    def __init__(self, input_dim, model_dim, class_weight, num_classes=2, num_heads=8, num_layer=6):
        self.model = CustomGRUContract(input_dim=input_dim, model_dim=model_dim, num_classes=num_classes,
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
            for batch_idx, (contracts, vectors, labels, masks, flag_labels) in enumerate(dataloader):
                vectors = vectors.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                flag_labels = flag_labels.to(device)

                self.optimizer.zero_grad()
                output = self.model(vectors, masks)

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
            for batch_idx, (contracts, vectors, labels, masks, flag_labels) in enumerate(dataloader):
                vectors = vectors.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                flag_labels = flag_labels.to(device)

                self.optimizer.zero_grad()
                output = self.model(vectors, masks)

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
