import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

#from dataset import TestDataset, MaskBaseDataset, TestDataset_noR
#from dataset_mike import *
from new_dataset_mike import *
from model import BaseModel, SwinTransformer384, VOLO_D5_224, ConvNext_Base, ResNest50, ResNext50


def load_model(saved_model, num_classes, device, model):
    model_cls = getattr(import_module("model"), model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_model2(model_class, num_classes):
    if model_class == 'ConvNext_Base':
        model = ConvNext_Base(num_classes=num_classes)
        model.load_state_dict(torch.load(os.path.join('./model/exp16/checkpoints/exp16.pth')))
    elif model_class == 'ResNest50':
        model = ResNest50(num_classes=num_classes)
        model.load_state_dict(torch.load(os.path.join('./model/exp17/checkpoints/exp17.pth')))
    elif model_class == 'ResNext50':
        model = ResNext50(num_classes=num_classes)
        model.load_state_dict(torch.load(os.path.join('./model/exp18/checkpoints/exp18.pth')))
    return model

@torch.no_grad()
def inference(data_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    # model1 = load_model('./model/exp16', num_classes, device, 'ConvNext_Base').to(device)
    # model2 = load_model('./model/exp17', num_classes, device, 'ResNest50').to(device)
    # model3 = load_model('./model/exp18', num_classes, device, 'ResNext50').to(device)

    model1 = load_model2('ConvNext_Base', 18).to(device)
    model2 = load_model2('ResNest50', 18).to(device)
    model3 = load_model2('ResNext50', 18).to(device)

    model1.eval()
    model2.eval()
    model3.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths=img_paths, resize=args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0, #multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            # average
            # images = images.to(device)
            # pred1 = model1(images)
            # #pred1 = pred1.argmax(dim=-1)
            # print('pred1:', pred1)
            
            # pred2 = model2(images)
            # #pred2 = pred2.argmax(dim=-1)
            # print('pred2:', pred2)

            # pred = (pred1 + pred2)/2 
            # pred = pred.argmax(dim=-1)
            # print('pred:', pred)
            # preds.extend(pred.cpu().numpy())

            # soft voting
            images = images.to(device)
            pred1 = model1(images)
            shape1 = pred1.argmax(dim=-1) # used for shape only
            #print('pred1:', pred1)
            
            pred2 = model2(images)
            #pred2 = pred2.argmax(dim=-1)
            #print('pred2:', pred2)

            pred3 = model3(images)
            #pred3 = pred3.argmax(dim=-1)
            #print('pred3:', pred3)

            pred = (pred1 + pred2 + pred3)/(shape1.shape[0]) 
            pred = pred.argmax(dim=-1)
            #print('pred:', pred)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(370, 324), help='resize size for image when you trained (default: (96, 128))')
    #parser.add_argument('--model', type=str, default='ResNest50', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    #parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp10'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    #model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, output_dir, args)
