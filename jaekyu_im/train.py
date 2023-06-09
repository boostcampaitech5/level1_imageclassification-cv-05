import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from dataset import MaskBaseDataset, BaseAugmentation, CustomAugmentation, BaseAugmentation_nonR
from model import BaseModel, SwinTransformer384, VOLO_D5_224, ConvNext_Base, ResNest50, ResNext50
from loss import FocalLoss, F1Loss, LabelSmoothingLoss
#from dataset_mike import *
from new_dataset_mike import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentations ********************************************************
    train_transform=Compose([
        RandomRotation((-15,15),Image.BILINEAR),
        MyCrop(), # 420,324
        RandomApply([RandomCrop(args.resize)],p=0.5), 
        RandomHorizontalFlip(0.5),
        Resize(args.resize,Image.BILINEAR),
        ToTensor(),
        Grayscale(3),
        RandomAutocontrast(p=1),
        RandomAdjustSharpness(sharpness_factor=4, p=1),
        GaussianNoise(mean=0, std=0.05, resize=args.resize),
        RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    
    val_transform=Compose([
        MyCrop(),
        Resize(args.resize,Image.BILINEAR),
        ToTensor(),
        Grayscale(3),
        RandomAutocontrast(p=1),
        RandomAdjustSharpness(sharpness_factor=4, p=1),
        GaussianNoise(mean=0, std=0.05, resize=args.resize),
        ])
    
    # -- data_set **************************************************************
    train_data = MaskBaseDataset(data_dir=data_dir, transform=train_transform, val_ratio = 0.1, mode='train_2')
    val_data = MaskBaseDataset(data_dir=data_dir, transform=val_transform, val_ratio=0.1, mode='val_2')  
    print("Total Train Data: ", len(train_data))
    print("Total Validation Data: ", len(val_data))

    # -- data_loader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=0,  # multiprocessing.cpu_count() // 2
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.valid_batch_size,
        num_workers=0,  # multiprocessing.cpu_count() // 2
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    # -- model *********************************************************************
    model = ResNext50(num_classes=18).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric *************************************************************
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.7)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
           
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                # if figure is None:
                #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #     inputs_np = MaskBaseDataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                #     figure = grid_image(inputs_np, labels, preds, n=16, shuffle=True)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_data)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")

            # checkpoints
            if os.path.isdir(os.path.join(save_dir, "checkpoints")) == False:
                os.mkdir(os.path.join(save_dir, "checkpoints"))
            torch.save(model.module.state_dict(), f"{save_dir}/checkpoints/Epoch:{epoch}, Val acc: {val_acc:4.2%}, Val loss: {val_loss:4.2%}")

            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            #logger.add_figure("results", figure, epoch)
            print()

        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--resize', nargs="+", type=tuple, default=(370, 324), help='resize size for image when training') # default [128, 96], 370, 324
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=8, help='input batch size for validing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-3(0.001)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--lr_decay_step', type=int, default=3, help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data_new/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
