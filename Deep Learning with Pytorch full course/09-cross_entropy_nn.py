import torch
import torch.nn as nn


# Binary class problem
class NeuralNetBinaryClass(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):  # num_classes is basically the output size
        super(NeuralNetBinaryClass, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred = torch.sigmoid(out)  # for the binary case i need to add the sigmoid
        return y_pred

    # Multiclass problem


class NeuralNetMultiClass(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):  # num_classes is basically the output size
        super(NeuralNetMultiClass, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out  # NB: no softmax at the end


model = NeuralNetMultiClass(input_size=28 * 28, hidden_size=5, num_classes=3)
loss_func = nn.CrossEntropyLoss()  # applies Softmax under the hood
