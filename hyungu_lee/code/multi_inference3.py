########인퍼런스 모델 input 확인, dataset resize확인, train size 확인####
import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import get_model
from dataset import TestDataset, MaskBaseDataset

from PIL import Image
from dataset import MyCrop
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
age_model_dir = '/opt/ml/log/multi_label/age/mike_60/00004.pt'
gender_model_dir = '/opt/ml/log/multi_label/gender/color_norm/00002.pt'
mask_model_dir = '/opt/ml/log/multi_label/mask/mike_aug/00002.pt'
@torch.no_grad()
def inference(data_dir, age_model_dir, gender_model_dir,mask_model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    
    age_model = get_model(classes=3,input_channel=3).cuda()
    age_model.load_state_dict(torch.load(age_model_dir))
    
    gender_model = get_model(classes=2,input_channel=3).cuda()
    gender_model.load_state_dict(torch.load(gender_model_dir))
    
    mask_model = get_model(classes=3,input_channel=3).cuda()
    mask_model.load_state_dict(torch.load(mask_model_dir))
    age_model.eval()
    gender_model.eval()
    mask_model.eval()
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    ####################transform############################
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
    
    dataset = TestDataset(img_paths, args.resize, transform = age_trans)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds_age = []
    preds_gender=[]
    preds_mask=[]
    preds=[]
    with torch.no_grad():
        for idx, images in enumerate(loader):
            
            images= images.to(device)
            
            
            
            pred = age_model(images)
            pred = pred.argmax(dim=-1)
            print('pred_age',pred[0])

            #print('mask', pred_mask.data)
            #print('gender', pred_gender.data)
            #print('age', pred_age.data)

            print('pred',pred)
            preds_age.extend(pred.cpu().numpy())
    dataset = TestDataset(img_paths, args.resize, transform = gender_trans)
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
            
            images= images.to(device)
            
            
            
            pred = gender_model(images)
            pred = pred.argmax(dim=-1)
            print('pred_gender',pred[0])

            #print('mask', pred_mask.data)
            #print('gender', pred_gender.data)
            #print('age', pred_age.data)


            #print('pred',pred)
            preds_gender.extend(pred.cpu().numpy())
    dataset = TestDataset(img_paths, args.resize, transform = mask_trans)
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
            
            images= images.to(device)
            
            
            
            pred = mask_model(images)
            pred = pred.argmax(dim=-1)
            print('pred_mask',pred[0])

            #print('mask', pred_mask.data)
            #print('gender', pred_gender.data)
            #print('age', pred_age.data)
            
            #pred = pred.clone()


            #print('pred',pred)
            preds_mask.extend(pred.cpu().numpy())
    for i,data in enumerate(zip(preds_age,preds_gender,preds_mask)):
        preds.append(MaskBaseDataset.encode_multi_class(preds_mask[i], preds_gender[i], preds_age[i]))

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output14.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/multi_label/three_model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, age_model_dir, gender_model_dir,mask_model_dir, output_dir, args)
