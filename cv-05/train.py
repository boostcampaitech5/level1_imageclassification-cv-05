import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from loss import *
from model import *

###
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold

from pytorchtools import EarlyStopping

import datetime
import pytz
import torch, gc
gc.collect()
torch.cuda.empty_cache()

### 저장 폴더명 생성
kst = pytz.timezone('Asia/Seoul')
now = datetime.datetime.now(kst)
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S')

from PIL import Image
import PIL


def getDataloader(dataset, train_idx, valid_idx, train_transformer, val_transformer, train_batch_size, val_batch_size, num_workers):
    dataset.set_transform(train_transformer)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    dataset.set_transform(val_transformer)
    val_set = torch.utils.data.Subset(dataset, valid_idx)
    print(f"train: {len(train_set)}, validation: {len(val_set)}")


    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=val_batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader

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

def increment_path(path, exist_ok=False):
    return f"{path}"

def trainSKFold(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    train_transformer = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        mode='train'
    )
    val_transformer = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        mode='val'
    )

    save_dir += '-' + args.model
    print(save_dir)

    skf = StratifiedKFold(n_splits=args.K)
    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, dataset.labels)):
        print("교차검증", i)
        wandb.init(project=folder_name + '-' + args.model, entity='ohsy0512')
        wandb.config.update(args)
        wandb.epochs = args.epochs
        wandb.run.name = folder_name + '-' + args.model + f'_{i}'

        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = np.inf

        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(num_classes=num_classes).to(device)

        train_loader, val_loader= getDataloader(dataset, train_idx, valid_idx,
        train_transformer, val_transformer, args.batch_size, args.valid_batch_size, 0)
        ### early stopping
        early_stopping = EarlyStopping(patience = args.early_stopping_num, verbose = True)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: FocalLoss
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-2
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.8)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        
        for epoch in range(args.epochs):
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

                ###
                labels_cp = labels.detach().clone().cpu().numpy()
                preds_cp = preds.detach().clone().cpu().numpy()

                train_precision = precision_score(labels_cp, preds_cp, average='macro')
                train_recall = recall_score(labels_cp, preds_cp, average='macro')
                train_f1 = f1_score(labels_cp, preds_cp, average='macro')
                
                
                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()                


                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    
                    ###
                    print(
                        f"Epoch[{epoch}/{args.epochs-1}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || "
                        f"precision : {train_precision:5.3%}, recall : {train_recall:5.3%}, f1 : {train_f1:5.3%}"
                    )
                    
                    ###
                    wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss, 'train_f1': train_f1})
                    
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
                
                ###
                val_precision_items = []
                val_recall_items = []
                val_f1_items = []
                val_labels = np.array([])
                val_preds = np.array([])
                
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    ###
                    labels_cp = labels.detach().clone().cpu().numpy()
                    preds_cp = preds.detach().clone().cpu().numpy()
                    val_labels = np.append(val_labels, labels_cp)
                    val_preds = np.append(val_preds, preds_cp)
                    
                    ###
                    valid_precision = precision_score(labels_cp, preds_cp, average='macro')
                    valid_recall = recall_score(labels_cp, preds_cp, average='macro')
                    valid_f1 = f1_score(labels_cp, preds_cp, average='macro')
                    val_precision_items.append(valid_precision)
                    val_recall_items.append(valid_recall)
                    val_f1_items.append(valid_f1)
                    
                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                
                ###
                val_precision = np.sum(val_precision_items) / len(val_loader)
                val_recall = np.sum(val_recall_items) / len(val_loader)
                val_f1 = np.sum(val_f1_items) / len(val_loader)
                print(classification_report(val_labels, val_preds))

                best_val_loss = min(best_val_loss, val_loss)

                ###
                if val_f1 > best_val_f1:
                    print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                    torch.save(model.state_dict(), f"{save_dir}/best{i}.pth")
                    best_val_f1 = val_f1
                torch.save(model.state_dict(), f"{save_dir}/last{i}.pth")
                
                ###
                print(
                    f"[Val] acc : {val_acc:5.3%}, loss: {val_loss:5.3} || "
                    f"precision : {val_precision:5.3%}, recall : {val_recall:5.3%}, f1 : {val_f1:5.3%} || "
                    f"best f1 : {best_val_f1:5.3%}, best loss: {best_val_loss:5.3}"
                )
                
                ###
                wandb.log({'val_acc': val_acc, 'val_loss': val_loss, 'val_f1':val_f1})
                print()
            logger.close()

            ###
            if args.early_stopping == True:
                early_stopping(val_loss, model) # 현재 과적합 상황 추적
                if early_stopping.early_stop: # 조건 만족 시 조기 종료
                    break
        wandb.finish()

def trainMike(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    train_transformer = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        mode='train'
    )
    val_transformer = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        mode='val'
    )
    save_dir += '-' + args.model
    print(save_dir)

    for i in range(args.K):
        wandb.init(project='Mike-SK-Fold-' + folder_name, entity='ohsy0512')
        wandb.config.update(args)
        wandb.epochs = args.epochs
        wandb.run.name = folder_name + '-' + args.model + f'_{i}'

        train_set = MikeBaseDataset(data_dir ='/opt/ml/input/data/train/images', transform = train_transformer, mode='train', idx = i, val_ratio=1/args.K)#이미지, 라벨
        val_set = MikeBaseDataset(data_dir='/opt/ml/input/data/train/images', transform = val_transformer, mode='val', idx = i, val_ratio = 1/args.K)

        # -- data_loader
        # train_set, val_set = dataset.split_dataset()

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)
        model = model_module(
            num_classes=18
        ).to(device)

        ###

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: FocalLoss
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-2
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.8)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        ###
        n_splits = 4
        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = np.inf
        skf = StratifiedKFold(n_splits=n_splits)
        
        # print(f"train: {len(train_set)}")
        print(f"train: {len(train_set)}, validation: {len(val_set)}")

        for epoch in range(args.epochs):
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

                ###
                labels_cp = labels.detach().clone().cpu().numpy()
                preds_cp = preds.detach().clone().cpu().numpy()

                train_precision = precision_score(labels_cp, preds_cp, average='macro')
                train_recall = recall_score(labels_cp, preds_cp, average='macro')
                train_f1 = f1_score(labels_cp, preds_cp, average='macro')
                
                
                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()                


                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    
                    ###
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || "
                        f"precision : {train_precision:5.3%}, recall : {train_recall:5.3%}, f1 : {train_f1:5.3%}"
                    )
                    
                    ###
                    wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss, 'train_f1':train_f1})
                    
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
                
                ###
                val_precision_items = []
                val_recall_items = []
                val_f1_items = []
                val_labels = np.array([])
                val_preds = np.array([])
                
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    ###
                    labels_cp = labels.detach().clone().cpu().numpy()
                    preds_cp = preds.detach().clone().cpu().numpy()
                    val_labels = np.append(val_labels, labels_cp)
                    val_preds = np.append(val_preds, preds_cp)
                    
                    ###
                    valid_precision = precision_score(labels_cp, preds_cp, average='macro')
                    valid_recall = recall_score(labels_cp, preds_cp, average='macro')
                    valid_f1 = f1_score(labels_cp, preds_cp, average='macro')
                    val_precision_items.append(valid_precision)
                    val_recall_items.append(valid_recall)
                    val_f1_items.append(valid_f1)
                    
                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                
                ###
                val_precision = np.sum(val_precision_items) / len(val_loader)
                val_recall = np.sum(val_recall_items) / len(val_loader)
                val_f1 = np.sum(val_f1_items) / len(val_loader)
                print(classification_report(val_labels, val_preds))
                
                best_val_loss = min(best_val_loss, val_loss)

                ###
                if val_f1 > best_val_f1:
                    print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                    torch.save(model.state_dict(), f"{save_dir}/best{i}.pth")
                    best_val_f1 = val_f1
                torch.save(model.state_dict(), f"{save_dir}/last{i}.pth")
                
                ###
                print(
                    f"[Val] acc : {val_acc:5.3%}, loss: {val_loss:5.3} || "
                    f"precision : {val_precision:5.3%}, recall : {val_recall:5.3%}, f1 : {val_f1:5.3%} || "
                    f"best f1 : {best_val_f1:5.3%}, best loss: {best_val_loss:5.3}"
                )
                
                ###
                wandb.log({'val_acc': val_acc, 'val_loss': val_loss, 'val_f1':val_f1})
                print()
            logger.close()

            ###
            if args.early_stopping == True:
                early_stopping(val_loss, model) # 현재 과적합 상황 추적
                if early_stopping.early_stop: # 조건 만족 시 조기 종료
                    break
        wandb.finish()
            
def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    train_transformer = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        mode='train'
    )
    val_transformer = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
        mode='val'
    )

    # -- save directory
    save_dir += '-' + args.model
    print(save_dir)

    wandb.init(project='mask-classification', entity='ohsy0512')
    wandb.config.update(args)
    wandb.epochs = args.epochs
    wandb.run.name = folder_name + '-' + args.model

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    train_set.dataset.transform = train_transformer
    val_set.dataset.transform = val_transformer


    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes=18
    ).to(device)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: FocalLoss
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.8)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    ###
    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf
    
    print(f"train: {len(train_set)}, validation: {len(val_set)}")

    for epoch in range(args.epochs):
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

            ###
            labels_cp = labels.detach().clone().cpu().numpy()
            preds_cp = preds.detach().clone().cpu().numpy()

            train_precision = precision_score(labels_cp, preds_cp, average='macro')
            train_recall = recall_score(labels_cp, preds_cp, average='macro')
            train_f1 = f1_score(labels_cp, preds_cp, average='macro')
            
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()                


            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                
                ###
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || "
                    f"precision : {train_precision:5.3%}, recall : {train_recall:5.3%}, f1 : {train_f1:5.3%}"
                )
                
                ###
                wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss, 'train_f1':train_f1})
                
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
            
            ###
            val_precision_items = []
            val_recall_items = []
            val_f1_items = []
            val_labels = np.array([])
            val_preds = np.array([])
            
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                ###
                labels_cp = labels.detach().clone().cpu().numpy()
                preds_cp = preds.detach().clone().cpu().numpy()
                val_labels = np.append(val_labels, labels_cp)
                val_preds = np.append(val_preds, preds_cp)
                
                ###
                valid_precision = precision_score(labels_cp, preds_cp, average='macro')
                valid_recall = recall_score(labels_cp, preds_cp, average='macro')
                valid_f1 = f1_score(labels_cp, preds_cp, average='macro')
                val_precision_items.append(valid_precision)
                val_recall_items.append(valid_recall)
                val_f1_items.append(valid_f1)
                
                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            
            ###
            val_precision = np.sum(val_precision_items) / len(val_loader)
            val_recall = np.sum(val_recall_items) / len(val_loader)
            val_f1 = np.sum(val_f1_items) / len(val_loader)
            print(classification_report(val_labels, val_preds))
            
            best_val_loss = min(best_val_loss, val_loss)

            ###
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                torch.save(model.state_dict(), f"{save_dir}/best{i}.pth")
                best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{save_dir}/last{i}.pth")
            
            ###
            print(
                f"[Val] acc : {val_acc:5.3%}, loss: {val_loss:5.3} || "
                f"precision : {val_precision:5.3%}, recall : {val_recall:5.3%}, f1 : {val_f1:5.3%} || "
                f"best f1 : {best_val_f1:5.3%}, best loss: {best_val_loss:5.3}"
            )
            
            ###
            wandb.log({'val_acc': val_acc, 'val_loss': val_loss, 'val_f1':val_f1})
            print()
        logger.close()

        ###
        if args.early_stopping == True:
            early_stopping(val_loss, model) # 현재 과적합 상황 추적
            if early_stopping.early_stop: # 조건 만족 시 조기 종료
                break
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default= None, help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='Resnext50', help='model type (default: Resnext50)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default=folder_name, help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--early_stopping', type=bool, default=False, help='early stopping (default: True)')
    parser.add_argument('--early_stopping_num', type=int, default=1, help='early stopping (default: True)')
    parser.add_argument('--cutmix', type=int, default=0, help='cutmix augmentation (default: 0)')
    parser.add_argument('--K', type=int, default=10, help='K value for SK-Fold')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/changed_data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    # parser.add_argument('--cutmix_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/augmented_images'))
    args = parser.parse_args()

    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    

    # train(data_dir, model_dir, args)
    # trainSKFold(data_dir, model_dir, args)
    trainMike(data_dir, model_dir, args)

