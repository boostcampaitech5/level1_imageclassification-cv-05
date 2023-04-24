import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.models = models.resnet152(pretrained=True)
        self.resnet152 = self.modify(self.models, num_classes)
        
    def forward(self, x):
        x = self.resnet152(x)
        return x  
    
    def modify(self, model, out_features):
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], out_features)
        nn.init.xavier_uniform_(model.fc.weight)

        stdv = 1 / math.sqrt(model.fc.weight.shape[1])
        model.fc.bias.data.uniform_(-stdv, stdv)

        for param in model.parameters():
            param.requires_grad = True
                 
        return model
    
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
    
class ResNest50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True) 

        # load pretrained models, using ResNeSt-50 as an example
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        fc_layer = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(2048, num_classes))

        model.fc = fc_layer
        self.model = model
        self.linear_param_initialize(self.model, 2048)

    #fc가중치 초기화
    def linear_param_initialize(self, model, n_feature):
        for module in model.fc.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                stdv = 1 / math.sqrt(n_feature) # n_feature = in_feature
                module.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = self.model(x)
        return output
    
class ResNext50(nn.Module):
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
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes))
        
        self.resnext50_32x4d.fc = fc3
        self.initialize_weights(self.resnext50_32x4d.fc)

    def forward(self, x):
        x = self.resnext50_32x4d(x)
        return x

    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
