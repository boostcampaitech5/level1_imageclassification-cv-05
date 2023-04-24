#from torchvision.models import efficientnet_v2_l
from torchsummary import summary
import torch
import os
import dataset
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, Grayscale, ToPILImage, RandomRotation, RandomCrop, RandomHorizontalFlip, RandomApply
import torchvision.transforms.functional as TF
from loss import FocalLoss, F1Loss, LabelSmoothingLoss
from model import get_model
from torchsummary import summary
from dataset import MaskBaseDataset, MyCrop
cuda = True if torch.cuda.is_available() else False

#############varaible#########
weight_path = '/opt/ml/log/multi_label/final/gender/gender_2'
log_path = '/opt/ml/log/multi_label/final/gender/gender_2'
train_data_mode = 'train_2'
val_data_mode = 'val_2'
val_ratio= 0.2
batch_size = 64
lr = 0.0001
classes = 2
train_mode = 'gender' #full, mask, gender, age

input_channel=3
mean=(0.55800916, 0.51224077, 0.47767341)
std=(0.21817792, 0.23804603, 0.25183411)
smoothing = 0.2
f1_w=0.0
ls_w=0.0
fo_w = 1.0
focal_weight = torch.tensor([1.,0.8]).cuda()

########model#####
#age
model = get_model(classes=classes,input_channel=input_channel)
model.cuda()

#######criterion###################

criterionfo =FocalLoss(weight = focal_weight)
criterionf1 =F1Loss(classes=classes)
criterionl =LabelSmoothingLoss(classes=classes,smoothing=smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
print(list(model.modules()))

############Data###############


train_transform=Compose([
                   Resize((370,324),Image.BILINEAR),
                   ToTensor(),
                   Normalize(mean=mean,std=std)
                   ])
val_transform=Compose([
                   Resize((370,324),Image.BILINEAR),
                   ToTensor(),
                   Normalize(mean=mean,std=std)
                    ])
train_data = MaskBaseDataset('/opt/ml/input/data/train/images',transform=train_transform,val_ratio=val_ratio, mode=train_data_mode)#이미지, 라벨
val_data = MaskBaseDataset('/opt/ml/input/data/train/images',transform=val_transform,val_ratio=val_ratio, mode=val_data_mode)
print(len(train_data))
print(len(val_data))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
#label to train label
def label2label(label):
    global train_mode
    a = ['mask', 'gender', 'age']
    if train_mode=='full':
        return None
    else :
        for i,d in enumerate(label.data):
            label.data[i] = dataset.MaskBaseDataset.decode_multi_class(d)[a.index(train_mode)]
    return None

# Misc
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Training
writer = SummaryWriter(log_path)

lens=len(train_data)//batch_size if len(train_data)%batch_size==0 else len(train_data)//batch_size + 1
print('step=',lens)
print(lens)

for epoch in range(0,30):
    model.train()
    for _iter, (img, label) in enumerate(train_loader):
        # optimizer에 저장된 미분값을 0으로 초기화
        optimizer.zero_grad()
        img = img.cuda()
        label2label(label)
        label = label.cuda()
        #print(label)
        #print(img.size())
        # 모델에 이미지 forward
        pred_logit = model(img)

        # loss 값 계산
        loss = f1_w*criterionf1(pred_logit, label) + ls_w*criterionl(pred_logit, label) + fo_w*criterionfo(pred_logit, label)
        writer.add_scalar("Loss/train", loss, epoch*lens+_iter+1)
        writer.flush()
        # Backpropagation
        loss.backward()
        optimizer.step()
        print('train loss : ',loss.data)
    model.eval()
    valid_loss, valid_acc = AverageMeter(), AverageMeter()
    for img, label in val_loader:
        
        img = img.cuda()
        label2label(label)
        label = label.cuda()
        with torch.no_grad():
          pred_logit = model(img)

        # loss 값 계산
        loss = f1_w*criterionf1(pred_logit, label) + ls_w*criterionl(pred_logit, label) + fo_w*criterionfo(pred_logit, label)

        # Accuracy 계산
        pred_label = torch.argmax(pred_logit, 1)
        acc = (pred_label == label).sum().item() / len(img)
        valid_loss.update(loss.item(), len(img))
        valid_acc.update(acc, len(img))
        
    valid_loss = valid_loss.avg
    valid_acc = valid_acc.avg
    print('epoch:',epoch,'   loss:',valid_loss,'    valid_acc:',valid_acc)
    writer.add_scalar("Loss/val",valid_loss, epoch)
    writer.add_scalar("acc/val", valid_acc, epoch)
    writer.flush()
    torch.save(model.state_dict(),os.path.join(weight_path,f"{epoch:05}.pt"))

