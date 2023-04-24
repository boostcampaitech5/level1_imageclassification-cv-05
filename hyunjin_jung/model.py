import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        fc_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=18))
            
        self.model.fc = fc_layer
        for module in fc_layer.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                stdv = 1 / math.sqrt(2048) # n_feature is in_feature
                module.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)
