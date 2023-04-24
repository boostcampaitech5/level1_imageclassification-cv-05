import os
import torch
import numpy as np
import torch.nn.functional as F
import wandb
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from utils.Dataset import Class18Dataset, cfg, get_img_paths
from PIL import Image
from baseline.dataset_v2 import MaskBaseDataset


#train
def train(train_loader, val_loader, model, optimizer, criterion, scheduler, model_name, epochs=30, batch_size=32, device='cuda'):

    wandb.init(
    project=f'mask_classification_{model_name}',
    name = f'{model_name}',
    config={
        "architecture": model_name,
        "dataset": "maskdataset",
        "batch_size" : batch_size
        })
        
    best_val_acc = 0
    best_val_loss = np.inf
    
    for epoch in tqdm(range(epochs),leave=False):
        print(f'\n Epoch:{epoch+1}'+'#'*30)
        c_lr = optimizer.param_groups[0]['lr']
        print('\n'+f'Current_lr:{c_lr}'+'\n')
        model.train()
        train_acc = 0
        train_loss = 0
        
        for b_idx, (imgs,labels) in tqdm(enumerate(train_loader),leave=False,          total=len(train_loader), ascii=True):
            
            #loss계산
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            
            if isinstance(criterion, dict):
                loss1 = 0.4*criterion['focal'](output, labels)
                loss2 = 0.4*criterion['label'](output, labels)
                loss3 = 0.2*criterion['f1'](output, labels)
                loss = loss1 + loss2 + loss3
                
            else:
                loss = criterion(output, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            #acc계산
            pred = torch.max(output, dim=-1)[1]
            acc = torch.eq(pred, labels).sum().item()
        
            train_loss += loss
            train_acc += acc
        
            if (b_idx+1)%(len(train_loader)//10) == 0:
                print('Train_loss:{:.4f} \t Train_acc:{:.2f}%'\
                  .format(train_loss/(b_idx+1), train_acc/((b_idx+1)*batch_size)*100))
              
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        train_metrics={"train/train_loss": train_loss, "train/train_acc": train_acc*100}
        print('Epoch:{} \t 최종-Train_loss:{:.4f} \t 최종-Train_acc:{:.2f}%'\
                  .format(epoch+1, train_loss, train_acc*100))
        #lr update
        scheduler.step()
        
        #validation
        if (epoch+1)%1 == 0:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_acc = 0
                pred_list = torch.tensor([]).to(device)
                label_list = torch.tensor([]).to(device)
                for imgs, labels in tqdm(val_loader, leave=False, ascii=True):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    
                    if isinstance(criterion, dict):
                        loss1 = 0.4*criterion['focal'](output, labels)
                        loss2 = 0.4*criterion['label'](output, labels)
                        loss3 = 0.2*criterion['f1'](output, labels)
                        loss = loss1 + loss2 + loss3
                    else:
                        loss = criterion(output, labels)
        
                    #acc계산
                    pred = torch.max(output, dim=-1)[1]
                    
                    pred_list = torch.cat([pred_list,pred], dim=-1)
                    label_list = torch.cat([label_list,labels], dim=-1)
                    
                    acc = torch.eq(pred, labels).sum().item()
                    
                    val_loss += loss
                    val_acc += acc
                
                val_loss /= len(val_loader)
                val_acc /= len(val_loader.dataset)
                val_f1 = f1_score(label_list.cpu(), pred_list.cpu(), average='macro')
                
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    save_dir = os.path.join('/opt/ml/model/logs/parameters', model_name) 
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), 
                       os.path.join(save_dir,f'Epoch:{epoch+1}-loss:{val_loss:.4f}-acc:{val_acc*100:.2f}%-best.pt'))
                    best_val_acc = val_acc
                
            
            print('Epoch:{} \t Val_loss:{:.4f} \t Val_acc:{:.2f}% \t Val_f1:{:.4f}'\
                  .format(epoch+1 ,val_loss, val_acc*100, val_f1))
            val_metrics = {"val/val_loss": val_loss, "val/val_acc": val_acc*100, 
                          "val/val_f1": val_f1, "val/lr":c_lr}
            wandb.log({**train_metrics, **val_metrics}, step=epoch)
            
    wandb.finish()
   
        
        
        
        
#KFold-Train
def skfold_train(model, optimizer, criterion, scheduler, model_name, img_paths, cfg, train_transformer, val_transformer, epochs=30, batch_size=32, device='cuda'):
               
        
        
    wandb.init(

    project= 'mask classification', 
    name=f"{model_name}", 
    config={
        "architecture": model_name,
        "dataset": "maskdataset",
        "batch_size" : batch_size
            })
    
    best_val_acc = 0
    best_val_loss = np.inf
    
    for epoch in tqdm(range(epochs),leave=False):
        
        
        
        train_dataset = Class18Dataset(img_paths, cfg, transform=train_transformer)
        val_dataset = Class18Dataset(img_paths, cfg, transform=val_transformer)
    
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    
        x = np.array(train_dataset.img)
        y = np.array(train_dataset.label)

        for f_idx, (train_idx, test_idx) in tqdm(enumerate(skf.split(x,y)), total=4,\
                                               leave=False):
            print(f'\n Epoch:{epoch+1} KFold:{f_idx}'+'#'*30+'\n')
            c_lr = optimizer.param_groups[0]['lr']
            print('\n'+f'Current_lr:{c_lr}'+'\n')
            
            train_dataset.img = x[np.array(train_idx)]
            train_dataset.label = y[np.array(train_idx)]

            val_dataset.img = x[np.array(test_idx)]
            val_dataset.label = y[np.array(test_idx)]
        
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,\
                                      num_workers=1)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,\
                                    num_workers=1)
        
            model.train()
            train_acc = 0
            train_loss = 0
            
            for b_idx, (imgs,labels) in tqdm(enumerate(train_loader),leave=False,          total=len(train_loader), ascii=True):
                model.train()
                #loss계산
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
            
                if isinstance(criterion, dict):
                    loss1 = 0.8*criterion['focal'](output, labels)
                    loss2 = 0.2*criterion['f1'](output, labels)
                    loss = loss1 + loss2 
                
                else:
                    loss = criterion(output, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                #acc계산
                pred = F.softmax(output)
                pred = torch.max(pred, dim=-1)[1]
                acc = torch.eq(pred, labels).sum().item()
        
                train_loss += loss
                train_acc += acc
        
                if (b_idx+1)%(len(train_loader)//10) == 0:
                    print('KFold:{}\t Train_loss:{:.4f} \t Train_acc:{:.2f}%'\
                      .format(f_idx+1, train_loss/(b_idx+1), train_acc/((b_idx+1)*batch_size)*100))

            train_loss /= len(train_loader)
            train_acc /= len(train_loader.dataset)
            train_metrics = {"train/train_loss": train_loss, "train/train_acc": train_acc*100}
            print('Epoch:{} \t KFold{}최종-Train_loss:{:.4f} \t KFold{}최종-Train_acc:{:.2f}%'\
                  .format(epoch+1, f_idx+1, train_loss, f_idx+1, train_acc*100))
            #lr update
            scheduler.step()
        
            #validation
            if (f_idx+1)%1 == 0:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    val_acc = 0
                    pred_list = torch.tensor([]).to(device)
                    label_list = torch.tensor([]).to(device)
                    for imgs, labels in tqdm(val_loader, leave=False, ascii=True):
                        imgs, labels = imgs.to(device), labels.to(device)
                        output = model(imgs)
                    
                        if isinstance(criterion, dict):
                            loss1 = 0.8*criterion['focal'](output, labels)
                            loss2 = 0.2*criterion['f1'](output, labels)
                            loss = loss1 + loss2 
                        else:
                            loss = criterion(output, labels)
        
                        #acc계산
                        pred = F.softmax(output)
                        pred = torch.max(pred, dim=-1)[1]
                    
                        pred_list = torch.cat([pred_list,pred], dim=-1)
                        label_list = torch.cat([label_list,labels], dim=-1)
                    
                        acc = torch.eq(pred, labels).sum().item()
                    
                        val_loss += loss
                        val_acc += acc
                
                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader.dataset)
                    val_f1 = f1_score(label_list.cpu(), pred_list.cpu(), average='macro')
                    
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        save_dir = os.path.join('/opt/ml/model/logs/parameters', model_name) 
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.state_dict(), 
                       os.path.join(save_dir,f'Epoch{epoch+1}_KFold{f_idx+1}-loss:{val_loss:.4f}-acc:{val_acc*100:.2f}%-best.pt'))
                        best_val_acc = val_acc
                    
                print('Epoch:{} \t KFold{}_Val_loss:{:.4f} \t KFold{}Val_acc:{:.2f}% \t KFold{}_Val_f1:{:.4f}'\
                  .format(epoch+1 , f_idx+1, val_loss, f_idx+1, val_acc*100, f_idx+1, val_f1))
                val_metrics = {"val/val_loss": val_loss, "val/val_acc": val_acc*100, 
                          "val/val_f1": val_f1}
                wandb.log({**train_metrics, **val_metrics})
            
    
    wandb.finish()
    

    
        
#KFold-Train
def mike_skfold_train(model, optimizer, criterion, scheduler, model_name, train_transformer, val_transformer, epochs=30, batch_size=32, device='cuda'):
               
        
        
    wandb.init(

    project= 'mask classification', 
    name=f"{model_name}", 
    config={
        "architecture": model_name,
        "dataset": "maskdataset",
        "batch_size" : batch_size
            })
    
    best_val_acc = 0
    best_val_loss = np.inf
    train_mode = ['train_0','train_1','train_2','train_3','train_4']
    val_mode = ['val_0','val_1','val_2','val_3','val_4']
    
    for epoch in tqdm(range(epochs),leave=False):
        
        
        for f_idx in tqdm(range(5),leave=False):
            print(f'\n Epoch:{epoch+1} KFold:{f_idx+1}'+'#'*30+'\n')
            c_lr = optimizer.param_groups[0]['lr']
            print('\n'+f'Current_lr:{c_lr}'+'\n')
            
        
        
            train_dataset = MaskBaseDataset('/opt/ml/data/train/images',
                                            transform=train_transformer, mode=train_mode[f_idx] ,val_ratio=0.2)
            val_dataset = MaskBaseDataset('/opt/ml/data/train/images', 
                                          transform=val_transformer, mode=val_mode[f_idx], val_ratio=0.2)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
                                
            model.train()
            train_acc = 0
            train_loss = 0
            
            for b_idx, (imgs,labels) in tqdm(enumerate(train_loader),leave=False,          total=len(train_loader), ascii=True):
                
                #loss계산
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
            
                if isinstance(criterion, dict):
                    loss1 = 0.8*criterion['focal'](output, labels)
                    loss2 = 0.2*criterion['f1'](output, labels)
                    loss = loss1 + loss2 
                
                else:
                    loss = criterion(output, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                #acc계산
                pred = F.softmax(output)
                pred = torch.max(pred, dim=-1)[1]
                acc = torch.eq(pred, labels).sum().item()
        
                train_loss += loss
                train_acc += acc
        
                if (b_idx+1)%(len(train_loader)//10) == 0:
                    print('KFold:{}\t Train_loss:{:.4f} \t Train_acc:{:.2f}%'\
                      .format(f_idx+1, train_loss/(b_idx+1), train_acc/((b_idx+1)*batch_size)*100))

            train_loss /= len(train_loader)
            train_acc /= len(train_loader.dataset)
            train_metrics = {"train/train_loss": train_loss, "train/train_acc": train_acc*100}
            print('Epoch:{} \t KFold{}최종-Train_loss:{:.4f} \t KFold{}최종-Train_acc:{:.2f}%'\
                  .format(epoch+1, f_idx+1, train_loss, f_idx+1, train_acc*100))
            #lr update
            scheduler.step()
        
            #validation
            if (f_idx+1)%1 == 0:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    val_acc = 0
                    pred_list = torch.tensor([]).to(device)
                    label_list = torch.tensor([]).to(device)
                    for imgs, labels in tqdm(val_loader, leave=False, ascii=True):
                        imgs, labels = imgs.to(device), labels.to(device)
                        output = model(imgs)
                    
                        if isinstance(criterion, dict):
                            loss1 = 0.8*criterion['focal'](output, labels)
                            loss2 = 0.2*criterion['f1'](output, labels)
                            loss = loss1 + loss2 
                        else:
                            loss = criterion(output, labels)
        
                        #acc계산
                        pred = F.softmax(output)
                        pred = torch.max(pred, dim=-1)[1]
                    
                        pred_list = torch.cat([pred_list,pred], dim=-1)
                        label_list = torch.cat([label_list,labels], dim=-1)
                    
                        acc = torch.eq(pred, labels).sum().item()
                    
                        val_loss += loss
                        val_acc += acc
                
                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader.dataset)
                    val_f1 = f1_score(label_list.cpu(), pred_list.cpu(), average='macro')
                    
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        save_dir = os.path.join('/opt/ml/model/logs/parameters', model_name) 
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.state_dict(), 
                       os.path.join(save_dir,f'Epoch{epoch+1}_KFold{f_idx+1}-loss:{val_loss:.4f}-acc:{val_acc*100:.2f}%-best.pt'))
                        best_val_acc = val_acc
                    
                print('Epoch:{} \t KFold{}_Val_loss:{:.4f} \t KFold{}Val_acc:{:.2f}% \t KFold{}_Val_f1:{:.4f}'\
                  .format(epoch+1 , f_idx+1, val_loss, f_idx+1, val_acc*100, f_idx+1, val_f1))
                val_metrics = {"val/val_loss": val_loss, "val/val_acc": val_acc*100, 
                          "val/val_f1": val_f1}
                wandb.log({**train_metrics, **val_metrics})
            
    
    wandb.finish()