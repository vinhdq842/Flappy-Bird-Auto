import torch
from torch import nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    def __init__(self, n_classes=2, n_temp_frames=4):
        super().__init__()
        # input bs x 4 x 84 x 84
        self.conv1 = nn.Conv2d(n_temp_frames, 32, kernel_size=(7, 7), stride=(3, 3))
        # output1 bs x 32 x 26 x 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        # output2 bs x 64 x 11 x 11
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        # output3 bs x 64 x 9 x 9

        self.fc1 = nn.Linear(9 * 9 * 64, 512)
        self.inter = nn.Conv1d(512, 512, kernel_size=1)
        self.fc2 = nn.Linear(512, 256)
        self.final = nn.Conv1d(256, n_classes, kernel_size=1)

        self.bn = nn.BatchNorm2d(n_temp_frames)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.conv1(self.bn(x))))
        x = self.dropout1(torch.relu(self.conv2(x)))
        x = self.dropout1(F.gelu(self.conv3(x)))

        x = self.dropout2(torch.relu(self.fc1(x.view(x.size(0), -1))))
        x = x.unsqueeze(-1)
        x = self.dropout2(torch.relu(self.inter(x)))

        x = x.squeeze(-1)
        x = self.dropout1(F.gelu(self.fc2(x)))
        x = x.unsqueeze(-1)

        return self.final(x).squeeze(-1)
