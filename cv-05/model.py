import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init
import torch

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    
class Resnext50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

        fc1 = nn.Sequential(nn.Linear(2048, num_classes))
        fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes))
        fc3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes))

        self.resnext50_32x4d.fc = fc2
        initialize_weights(self.resnext50_32x4d.fc)

    def forward(self, x):
        x = self.resnext50_32x4d(x)
        return x


def Resnest50(classes=18):
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

    fc_layer = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, classes))

    model.fc = fc_layer
    def linear_param_initialize(model, n_feature):
        for module in model.fc.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                stdv = 1 / math.sqrt(2048) # n_feature is in_feature
                module.bias.data.uniform_(-stdv, stdv)

    linear_param_initialize(model, 2048)
    return model