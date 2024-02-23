import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os


class MRIImaging3DConvModel(nn.Module):
    def __init__(self, nClass):
        super(MRIImaging3DConvModel, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=8)
        self.bn2 = nn.BatchNorm3d(num_features=16)
        self.bn3 = nn.BatchNorm3d(num_features=32)
        self.bn4 = nn.BatchNorm3d(num_features=64)
        self.bn5 = nn.BatchNorm3d(num_features=128)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dp = nn.Dropout(0.5)
        self.dense1 = nn.Linear(19200, 1024)
        self.dense2 = nn.Linear(1024, 128)
        self.classifier = nn.Linear(128, nClass)

    def forward(self, inputs):
        # print(inputs.shape)
        x1 = F.relu(self.bn1(self.conv1(inputs)))
        x1 = self.pool(x1)
        # print(x1.shape)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.pool(x2)
        # print(x2.shape)
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = self.pool(x3)
        # print(x3.shape)
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4 = self.pool(x4)
        # print(x4.shape)
        x = F.relu(self.bn5(self.conv5(x4)))
        # print(x.shape)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        # print(x.shape)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.classifier(x)
        # print(x.shape) # (8,2)
        return x

    def extract_embedding(self, inputs):
        x1 = F.relu(self.bn1(self.conv1(inputs)))
        x1 = self.pool(x1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.pool(x2)
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = self.pool(x3)
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4 = self.pool(x4)
        x = F.relu(self.bn5(self.conv5(x4)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x

    def calculate_gradients(self, x, y):
        x = x.float()
        x.requires_grad = True
        prediction = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(prediction, y)
        loss.backward()

        gradients = x.grad
        return gradients