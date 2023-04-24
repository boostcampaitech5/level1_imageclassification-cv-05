import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from resnest.torch import resnest50


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


class ThreeOneNet(nn.Module):
    def __init__(self, maskmodel, gendermodel, agemodel):
        super().__init__()
        self.maskmodel = maskmodel
        self.gendermodel = gendermodel
        self.agemodel = agemodel
        
        self.fc = nn.Sequential(
                    nn.Linear(in_features=8, out_features=128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=128, out_features=18)
                )
        
    def forward(self, x):
        x1 = self.maskmodel(x)
        x2 = self.gendermodel(x)
        x3 = self.agemodel(x)
        x = torch.cat([x1,x2,x3], dim=-1)
        x = self.fc(x)
        return x
    
    

class Resnext50(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

        fc1 = nn.Sequential(nn.Linear(2048, num_classes))
        fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes))

        self.resnext50_32x4d.fc = fc2
        initialize_weights(self.resnext50_32x4d.fc)

    def forward(self, x):
        x = self.resnext50_32x4d(x)
        return x

    
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
            

    
class Resnest50(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.resnest50 = resnest50(pretrained=True)

        fc1 = nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_features=2048, out_features=18)
        )
        
        fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes))

        self.resnest50.fc = fc1
        initialize_weights(self.resnest50, 2048)
        

    def forward(self, x):
        x = self.resnest50(x)
        return x


def initialize_weights(model, n_feature=2048):        
        
    for m in model.fc.children():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            stdv = 1 / math.sqrt(n_feature)
            m.bias.data.uniform_(-stdv, stdv)
            n_feature //= 2


class SwinTransformer384(nn.Module): 
    def __init__(self, backbone='swin_base_patch4_window12_384', pretrained=True):
        super(SwinTransformer384, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained) 
        self.backbone.reset_classifier(18)
        
        # Linear Xavier Weight Initialization
        nn.init.xavier_uniform_(self.backbone.head.weight)
        stdv = 1 / math.sqrt(self.backbone.head.weight.shape[1])
        self.backbone.head.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = self.backbone(x)                         
        return output
    
    

class VOLO_D5_224(nn.Module):
    def __init__(self):
        super(VOLO_D5_224, self).__init__()
        self.backbone = timm.create_model('volo_d5_224', pretrained=True)
        self.backbone.reset_classifier(18)
        
        # Linear Xavier Weight Initialization
        nn.init.xavier_uniform_(self.backbone.head.weight)
        stdv = 1 / math.sqrt(self.backbone.head.weight.shape[1])
        self.backbone.head.bias.data.uniform_(-stdv, stdv)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        output = self.backbone(x)                         
        return output
    
class ConvNext_Base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('convnext_base_in22ft1k', pretrained=True)
        self.backbone.reset_classifier(num_classes)
        
        # Linear Xavier Weight Initialization
        nn.init.xavier_uniform_(self.backbone.head.fc.weight)
        stdv = 1 / math.sqrt(self.backbone.head.fc.weight.shape[1])
        self.backbone.head.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = self.backbone(x)                         
        return output