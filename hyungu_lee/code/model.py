import torch
from torch import nn
from torchsummary import summary
import math
from torch.utils.tensorboard import SummaryWriter
# get list of models
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

# load pretrained models, using ResNeSt-50 as an example


def get_model(model='model',input_channel=3, classes=18):
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
    #model.conv1[0]= nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    return model
a = get_model()
#print(a)