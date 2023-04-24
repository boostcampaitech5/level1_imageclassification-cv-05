import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image 
from glob import glob


#config 설정
class cfg:
    data_dir = os.path.join(os.getcwd(), 'input/data/train' )
    img_dir = f'{data_dir}/images'
    df_path = f'{data_dir}/train_2.csv'
    img_names = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
             'mask4', 'mask5', 'normal']
    
    #class 구분
    wear = [0,1,2,3,4,5]
    incorrect= [6,7,8,9,10,11]
    notwear = [12,13,14,15,16,17]

    male = [0,1,2,6,7,8,12,13,14]
    female = [3,4,5,9,10,11,15,16,17]

    young = [0,3,6,9,12,15] #30미만
    mid = [1,4,7,10,13,16] # 30이상 60미만
    old = [2,5,8,11,14,17] #60이상

    
#경로 함수
def get_img_paths(cfg):
    df = pd.read_csv(cfg.df_path)
    img_ids = df.path.values
    return [os.path.join(cfg.img_dir, img_id) for img_id in img_ids]




#Dataset 구성
class Class18Dataset(Dataset):
    
    def __init__(self, img_paths, cfg, transform=None):    
        self.img_paths = img_paths
        self.cfg = cfg
        self.transform = transform
        
        self.img = []
        self.label = []
        
        #img id별로 파일을 불러와서 클래스 분류 후 라벨링 작업하기
        for img_id in img_paths:
            img_list = glob(os.path.join(img_id, '*'))
            for img in img_list:
                
                #img주소 list에 추가
                self.img.append(img)
                
                #경로, 단어 단위로 분리
                words = img.split('/')
                
                #성별 확인
                gender = words[-2].split('_')[1]
                
                if gender == 'female':
                    class_set = set(self.cfg.female)
                else:
                    class_set = set(self.cfg.male)
                    
                #마스크 착용 상태 구분
                mask_state = words[-1] 
                
                if mask_state.startswith('mask'):
                    class_set = class_set&set(self.cfg.wear)
    
                elif 'incorrect' in mask_state:
                    class_set = class_set&set(self.cfg.incorrect)
            
                else:
                    class_set = class_set&set(self.cfg.notwear)
                    
                #연령대 구분
                age = int(words[-2].split('_')[-1])
                
                if age<30:
                    class_set = class_set&set(self.cfg.young)
                    
                elif 30<=age and age<59:
                    class_set = class_set&set(self.cfg.mid)
                
                else:
                    class_set = class_set&set(self.cfg.old)
                
                self.label.append(list(class_set)[0])
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = Image.open(self.img[idx])
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float)
        
        label = self.label[idx]
        
        return img, label

    
    
    
    
class GenderDataset(Dataset):
    
    def __init__(self, img_paths, cfg, transform=None):    
        self.img_paths = img_paths
        self.cfg = cfg
        self.transform = transform
        
        self.img = []
        self.label = []
        
        #img id별로 파일을 불러와서 클래스 분류 후 라벨링 작업하기
        for img_id in img_paths:
            img_list = glob(os.path.join(img_id, '*'))
            for img in img_list:
                
                #img주소 list에 추가
                self.img.append(img)
                
                #경로, 단어 단위로 분리
                words = img.split('/')
                
                #성별 확인
                gender = words[-2].split('_')[1]
                
                if gender == 'female':
                    self.label.append(1)
                else:
                    self.label.append(0)
                    
                
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = np.array(Image.open(self.img[idx]))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float)
        
        label = self.label[idx]
        
        return img, label
    

    
    
    
    
#Dataset 구성
class MaskDataset(Dataset):
    
    def __init__(self, img_paths, cfg, transform=None):    
        self.img_paths = img_paths
        self.cfg = cfg
        self.transform = transform
        
        self.img = []
        self.label = []
        
        #img id별로 파일을 불러와서 클래스 분류 후 라벨링 작업하기
        for img_id in img_paths:
            img_list = glob(os.path.join(img_id, '*'))
            for img in img_list:
                
                #img주소 list에 추가
                self.img.append(img)
                
                #경로, 단어 단위로 분리
                words = img.split('/')
                    
                #마스크 착용 상태 구분
                mask_state = words[-1] 
                
                if mask_state.startswith('mask'):
                    self.label.append(0)
    
                elif 'incorrect' in mask_state:
                    self.label.append(1)
            
                else:
                    self.label.append(2)
                
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = np.array(Image.open(self.img[idx]))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float)
        
        label = self.label[idx]
        
        return img, label

    
    
#Dataset 구성
class AgeDataset(Dataset):
    
    def __init__(self, img_paths, cfg, transform=None):    
        self.img_paths = img_paths
        self.cfg = cfg
        self.transform = transform
        
        self.img = []
        self.label = []
        
        #img id별로 파일을 불러와서 클래스 분류 후 라벨링 작업하기
        for img_id in img_paths:
            img_list = glob(os.path.join(img_id, '*'))
            for img in img_list:
                
                #img주소 list에 추가
                self.img.append(img)
                
                #경로, 단어 단위로 분리
                words = img.split('/')
                    
                #연령대 구분
                age = int(words[-2].split('_')[-1])
                
                if age<30:
                    self.label.append(0)
                    
                elif 30<=age and age<59:
                    self.label.append(1)
                
                else:
                    self.label.append(2)
                

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = np.array(Image.open(self.img[idx]))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float)
        
        label = self.label[idx]
        
        return img, label
    
    


    
#Dataset 구성
class Class18Dataset_Crops(Dataset):
    
    def __init__(self, img_paths, cfg, transform1=None, transform2=None, transform3=None):    
        self.img_paths = img_paths
        self.cfg = cfg
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        
        self.img = []
        self.label = []
        
        #img id별로 파일을 불러와서 클래스 분류 후 라벨링 작업하기
        for img_id in img_paths:
            img_list = glob(os.path.join(img_id, '*'))
            for img in img_list:
                
                #img주소 list에 추가
                self.img.append(img)
                
                #경로, 단어 단위로 분리
                words = img.split('/')
                
                #성별 확인
                gender = words[-2].split('_')[1]
                
                if gender == 'female':
                    class_set = set(self.cfg.female)
                else:
                    class_set = set(self.cfg.male)
                    
                #마스크 착용 상태 구분
                mask_state = words[-1] 
                
                if mask_state.startswith('mask'):
                    class_set = class_set&set(self.cfg.wear)
    
                elif 'incorrect' in mask_state:
                    class_set = class_set&set(self.cfg.incorrect)
            
                else:
                    class_set = class_set&set(self.cfg.notwear)
                    
                #연령대 구분
                age = int(words[-2].split('_')[-1])
                
                if age<30:
                    class_set = class_set&set(self.cfg.young)
                    
                elif 30<=age and age<59:
                    class_set = class_set&set(self.cfg.mid)
                
                else:
                    class_set = class_set&set(self.cfg.old)
                
                self.label.append(list(class_set)[0])
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        img = Image.open(self.img[idx])
        
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        img3 = self.transform3(img)
        img = torch.cat([img1,img2,img3],dim=0)
        
        label = self.label[idx]
        
        return img, label
