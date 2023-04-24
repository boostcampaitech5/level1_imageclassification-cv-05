########인퍼런스 모델 input 확인, dataset resize확인, train size 확인####
import argparse
import multiprocessing
import os
from importlib import import_module
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import get_model
from dataset import TestDataset, MaskBaseDataset
from dataset import MyCrop
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, Grayscale, ToPILImage, RandomRotation, RandomCrop, RandomHorizontalFlip,RandomApply
#################################안씀#######################################
def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
#################################^^안씀^^#######################################

###모델 웨이트 경로 정의
age1_model_dir = '/opt/ml/log/multi_label/final/age/age_0/00001.pt'
age2_model_dir = '/opt/ml/log/multi_label/final/age/age_1/00000.pt'
age3_model_dir = '/opt/ml/log/multi_label/final/age/age_2/00002.pt'

gender1_model_dir = '/opt/ml/log/multi_label/final/gender/gender_0/00000.pt'
gender2_model_dir = '/opt/ml/log/multi_label/final/gender/gender_1/00001.pt'
gender3_model_dir = '/opt/ml/log/multi_label/final/gender/gender_2/00000.pt'

mask1_model_dir = '/opt/ml/log/multi_label/final/mask/mask_0/00001.pt'
mask2_model_dir = '/opt/ml/log/multi_label/final/mask/mask_1/00001.pt'
mask3_model_dir = '/opt/ml/log/multi_label/final/mask/mask_2/00000.pt'
@torch.no_grad()
def inference(data_dir, age_model_dir, gender_model_dir,mask_model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    
    age_model1 = get_model(classes=3,input_channel=3).cuda()
    age_model1.load_state_dict(torch.load(age1_model_dir))
    age_model2 = get_model(classes=3,input_channel=3).cuda()
    age_model2.load_state_dict(torch.load(age2_model_dir))
    age_model3 = get_model(classes=3,input_channel=3).cuda()
    age_model3.load_state_dict(torch.load(age3_model_dir))
    
    gender_model1 = get_model(classes=2,input_channel=3).cuda()
    gender_model1.load_state_dict(torch.load(gender1_model_dir))
    gender_model2 = get_model(classes=2,input_channel=3).cuda()
    gender_model2.load_state_dict(torch.load(gender2_model_dir))
    gender_model3 = get_model(classes=2,input_channel=3).cuda()
    gender_model3.load_state_dict(torch.load(gender3_model_dir))

    mask_model1 = get_model(classes=3,input_channel=3).cuda()
    mask_model1.load_state_dict(torch.load(mask1_model_dir))
    mask_model2 = get_model(classes=3,input_channel=3).cuda()
    mask_model2.load_state_dict(torch.load(mask2_model_dir))
    mask_model3 = get_model(classes=3,input_channel=3).cuda()
    mask_model3.load_state_dict(torch.load(mask3_model_dir))

    
    age_model1.eval()
    gender_model1.eval()
    mask_model1.eval()
    
    age_model2.eval()
    gender_model2.eval()
    mask_model2.eval()
    
    age_model3.eval()
    gender_model3.eval()
    mask_model3.eval()

    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    ########################
    age_trans = Compose([ 
                    Grayscale(3),
                    MyCrop(),
                    Resize((370, 324),Image.BILINEAR),
                    ToTensor(),
                    #Normalize(mean=0.55800916, std=0.21817792)
                    ])
    mean=(0.55800916, 0.51224077, 0.47767341)
    std=(0.21817792, 0.23804603, 0.25183411)
    gender_trans = Compose([
                   Resize((370,324),Image.BILINEAR),
                   ToTensor(),
                   Normalize(mean=mean,std=std)
                    ])
    mask_trans = Compose([ 
                    Grayscale(3),
                    MyCrop(),
                    Resize((370, 324),Image.BILINEAR),
                    ToTensor(),
                    #Normalize(mean=0.55800916, std=0.21817792)
                    ])
    #################################
    
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize,transform = gender_trans)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds_g = []
    preds_a = []
    preds_m = []
    preds=[]
    a=0
    with torch.no_grad():
        for idx, images in enumerate(loader):

            gender_images = images.to(device)
      

            
            pred_gender1 = gender_model1(gender_images)
            pred_gender2 = gender_model2(gender_images)
            pred_gender3 = gender_model3(gender_images)
            pred_gender = pred_gender1 + pred_gender2 + pred_gender3
            pred_gender = pred_gender.argmax(dim=-1)
            
            
            
            #print('mask', pred_mask.data)
            #print('gender', pred_gender.data)
            #print('age', pred_age.data)

                
            preds_g.extend(pred_gender.cpu().numpy())
    dataset = TestDataset(img_paths, args.resize,transform = age_trans)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    with torch.no_grad():
        for idx, images in enumerate(loader):
            age_images = images.to(device)
            
            pred_age1 = age_model1(age_images)
            pred_age2 = age_model2(age_images)
            pred_age3 = age_model3(age_images)
            pred_age = F.softmax(pred_age1, dim=0)+F.softmax(pred_age2, dim=0)+F.softmax(pred_age3, dim=0)
            pred_age = pred_age.argmax(dim=-1)

            #print('mask', pred_mask.data)
            #print('gender', pred_gender.data)
            #print('age', pred_age.data)

                
            preds_a.extend(pred_age.cpu().numpy())
    dataset = TestDataset(img_paths, args.resize,transform = mask_trans)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    with torch.no_grad():
        for idx, images in enumerate(loader):

            mask_images = images.to(device)

            pred_mask1 = mask_model1(mask_images)
            pred_mask2 = mask_model2(mask_images)
            pred_mask3 = mask_model3(mask_images)
            pred_mask = pred_mask1 + pred_mask2 + pred_mask3
            pred_mask = pred_mask.argmax(dim=-1)
            
            
            #print('mask', pred_mask.data)
            #print('gender', pred_gender.data)
            #print('age', pred_age.data)
            
                
            preds_m.extend(pred_mask.cpu().numpy())
    if a ==0:
            a+=1
    for i,data in enumerate(zip(preds_a,preds_g,preds_m)):
        preds.append(MaskBaseDataset.encode_multi_class(preds_m[i], preds_g[i], preds_a[i]))
    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output17.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(0, 0), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/multi_label/nine_model1'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    age_model_dir = ''
    gender_model_dir = ''
    mask_model_dir = ''
    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, age_model_dir, gender_model_dir,mask_model_dir,  output_dir, args)
