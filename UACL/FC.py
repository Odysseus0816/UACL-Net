import torch
import torch.nn as nn


class FC_Class(nn.Module):

    def __init__(self, in_feat=512, num_classes=4):
        super(FC_Class, self).__init__()
        self.convf_dim = in_feat
        self.out_dim = num_classes

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.fc_classifier1 = nn.Linear(in_feat, in_feat // 2)
        self.bn_fc1 = nn.BatchNorm1d(in_feat // 2)

        self.fc_classifier2 = nn.Linear(in_feat // 2, in_feat // 4)
        self.bn_fc2 = nn.BatchNorm1d(in_feat // 4)

        self.fc_classifier3 = nn.Linear(in_feat // 4, in_feat // 8)
        self.bn_fc3 = nn.BatchNorm1d(in_feat // 8)

        self.fc_classifier4 = nn.Linear(in_feat // 8, num_classes)

    def forward(self, x):
        x = self.fc_classifier1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_classifier2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_classifier3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_classifier4(x)
        x = torch.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    FC = FC_Class(in_feat=512, num_classes=3)
    input = torch.randn(8, 512)
    output = FC(input)
    print(output.shape)
