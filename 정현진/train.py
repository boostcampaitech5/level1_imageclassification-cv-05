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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataset import MikeDataset
from dataset import TMycrop_transformer, TMycropDown_transformer, TMycropMid_transformer
from dataset import VMycrop_transformer, VMycropDown_transformer, VMycropMid_transformer
from dataset import Ensemble
from torch.utils.tensorboard import SummaryWriter

import wandb

from dataset import MaskBaseDataset
from loss import create_criterion


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

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    train_dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = train_dataset.num_classes  # 18

    val_dataset = dataset_module(
        data_dir=data_dir,
    )


    #transform
    train_dataset.set_transform(TMycrop_transformer)
    val_dataset.set_transform(VMycrop_transformer)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    skf = StratifiedKFold(10)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset.profiles, train_dataset.pre_labels)):
        t_idx = []
        for idx in train_idx:
            t_idx.extend(range(idx, idx+7))
        v_idx = []
        for idx in val_idx:
            v_idx.extend(range(idx, idx+7))

        train_set = Subset(train_dataset, t_idx)
        val_set = Subset(val_dataset, v_idx)

        # -- data_loader
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
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        weights = torch.tensor([0.0004165652560834198,
                    0.0005394877106257007,
                    0.002459863910979232,
                    0.00032625374599569955,
                    0.0002981807067626763,
                    0.001885224807564291,
                    0.0018718600484659816,
                    0.0024892441111694892,
                    0.012097659521450886,
                    0.0014166618874551819,
                    0.001274609799184368,
                    0.00922394404105895,
                    0.0018718600484659816,
                    0.0024892441111694892,
                    0.012097659521450886,
                    0.0014166618874551819,
                    0.001274609799184368,
                    0.00922394404105895]).to(device)
        criterion = create_criterion(args.criterion, weight = weights)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.7)


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
                        f"Fold[{fold}/10] || "
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar(f"Train/loss[{fold}]", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar(f"Train/accuracy[{fold}]", train_acc, epoch * len(train_loader) + idx)
                    wandb.log({f"Train/loss[{fold}]" : train_loss, f"Train/accuracy[{fold}]" : train_acc})

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            y_true = []
            y_pred = []
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
                    
                    y_true.extend(list(labels.cpu()))
                    y_pred.extend(list(preds.cpu()))

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        #inputs_np = dataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=8, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best{fold}.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last{fold}.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                wandb.log({f"Val/loss[{fold}]" : val_loss, f"Val/accuracy[{fold}]" : val_acc})
                wandb.log({f"plot[{fold}]" : wandb.Image(figure)})
                wandb.log({f"conf_mat[{fold}]" : wandb.sklearn.plot_confusion_matrix(y_true, y_pred)}, step = epoch)

                print()

def parse_args_from_config_file(config_file_path):
    with open(config_file_path) as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()

    for key, value in config.items():
        if isinstance(value, list):
            parser.add_argument(f"--{key}", nargs="+", type=type(value[0]), default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    return parser.parse_args()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # # Data and model checkpoints directories
    # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    # parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    # parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    # parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parse_args_from_config_file('config.json')
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    # Weight & Biases setting
    wandb.init(project="Mask_Classification", reinit=True)
    wandb.run.name = args.name
    wandb.run.save()
    wandb.config.update(args)

    train(data_dir, model_dir, args)
