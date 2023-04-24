import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
from torchvision import transforms
from glob import glob
import torchvision.transforms.functional as TF

### 사진에서 정면 전체, 얼굴, 옷 부분을 추출하는 코드입니다. ###

class MyCrop:
    def __init__(self,top=39, left=30, height=375, width=384-60):
        self.top = top
        self.left = left
        self.height = height
        self. width = width

    def __call__(self, x):
        return TF.crop(x, self.top, self.left, self.height, self.width)


class MyCropMid:
    def __init__(self,top=200, left=30, height=200, width=384-60):
        self.top = top
        self.left = left
        self.height = height
        self. width = width

    def __call__(self, x):
        return TF.crop(x, self.top, self.left, self.height, self.width)



class MyCropDown:
    def __init__(self,top=360, left=30, height=152, width=384-60):
        self.top = top
        self.left = left
        self.height = height
        self. width = width

    def __call__(self, x):
        return TF.crop(x, self.top, self.left, self.height, self.width)



class Ensemble:
    def __init__(self, transform1=None, transform2=None, transform3=None):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, img):

        img1 = self.transform1(img)
        img2 = self.transform2(img)
        img3 = self.transform3(img)

        img = torch.cat([img1,img2,img3],dim=0)

        return img

TMycrop_transformer=transforms.Compose([
                   transforms.RandomRotation((-15,15),Image.BILINEAR),
                   MyCrop(),
                   transforms.RandomApply([transforms.RandomCrop((355,324))],p=0.5),
                   transforms.Resize((355,324),Image.BILINEAR),
                   transforms.ToTensor(),
                   transforms.Grayscale(1),
                   transforms.RandomAutocontrast(p=1),
                   transforms.RandomAdjustSharpness(sharpness_factor=4, p=1)
                   ])

TMycropMid_transformer=transforms.Compose([
                   MyCropMid(),
                   transforms.Resize((355,324),Image.BILINEAR),
                   transforms.ToTensor(),
                   transforms.Grayscale(1),
                   transforms.RandomAutocontrast(p=1),
                   transforms.RandomAdjustSharpness(sharpness_factor=4, p=1)
                   ])

TMycropDown_transformer=transforms.Compose([
                   transforms.RandomRotation((-15,15),Image.BILINEAR),
                   MyCropDown(),
                   transforms.Resize((355,324),Image.BILINEAR),
                   transforms.ToTensor(),
                   transforms.Grayscale(1),
                   transforms.RandomAutocontrast(p=1),
                   transforms.RandomAdjustSharpness(sharpness_factor=4, p=1)
                   ])


#############################################################################


VMycrop_transformer=transforms.Compose([
                   MyCrop(),
                   transforms.Resize((355,324),Image.BILINEAR),
                   transforms.ToTensor(),
                   transforms.Grayscale(1),
                   transforms.RandomAutocontrast(p=1),
                   transforms.RandomAdjustSharpness(sharpness_factor=4, p=1)
                   ])

VMycropMid_transformer=transforms.Compose([
                   MyCropMid(),
                   transforms.Resize((355,324),Image.BILINEAR),
                   transforms.ToTensor(),
                   transforms.Grayscale(1),
                   transforms.RandomAutocontrast(p=1),
                   transforms.RandomAdjustSharpness(sharpness_factor=4, p=1)
                   ])

VMycropDown_transformer=transforms.Compose([
                   MyCropDown(),
                   transforms.Resize((355,324),Image.BILINEAR),
                   transforms.ToTensor(),
                   transforms.Grayscale(1),
                   transforms.RandomAutocontrast(p=1),
                   transforms.RandomAdjustSharpness(sharpness_factor=4, p=1)
                   ])

def train_transformer(img):
    return Ensemble(transform1=TMycrop_transformer, transform2=TMycropMid_transformer, transform3=TMycropDown_transformer)(img)

def val_transformer(img):
    return Ensemble(transform1=VMycrop_transformer, transform2=VMycropMid_transformer, transform3=VMycropDown_transformer)(img)