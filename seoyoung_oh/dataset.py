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
import transform
import copy
IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class MyCrop:
    """Rotate by one of the given angles."""

    def __init__(self,top=512-473, left=30, height=420, width=384-60):
        self.top = top
        self.left = left
        self.height = height
        self. width = width

    def __call__(self, x):
        return TF.crop(x, self.top, self.left, self.height, self.width)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, mode, **args):
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation((-15,15),Image.BILINEAR),
                MyCrop(),
                transforms.RandomApply([transforms.RandomCrop((370,324))],p=0.5),#(h,w)
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((370,324),Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Grayscale(3)
                ])
        elif mode == 'val':
            self.transform = transforms.Compose([
                MyCrop(),
                transforms.Resize((370,324),Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Grayscale(3)
                ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 59:
            return cls.MIDDLE
        else:
            return cls.OLD


class CutMixDataset(Dataset):
    num_classes = 3 * 2 * 3

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        
        self.image_path = glob(data_dir+'/*')
        self.file_name = os.listdir(data_dir)
        self.transform = None
        self.calc_statistics()

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        image = Image.open(self.image_path[index])
        file_name = self.file_name[index].split('.')[0]
        cutmix, index, left, right= file_name.split('_')
        left, right = int(left), int(right)
        multi_class_label = (torch.tensor(left), torch.tensor(right))
        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_path)


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.labels.append(mask_label * 6 + gender_label * 3 + age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MikeBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "mask6": MaskLabels.MASK,
        "mask7": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "incorrect_mask1": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
        "normal1": MaskLabels.NORMAL
    }

    def __init__(self, data_dir, transform=None, mode='train', idx=0, val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mode =mode
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.data_label_distribution=defaultdict(int)
        self.data_label_paths= defaultdict(list)
        self.seed = idx
        self.setup()
        self.transform = transform

    def setup(self):
        # self.seed = int(self.mode.split('_')[1])
        
        profiles = os.listdir(self.data_dir)
        profiles.sort()
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue
            id, gender, race, age = profile.split("_")
            self.data_label_distribution[str(gender + age)] +=1
            self.data_label_paths[str(gender + age)].append(profile)
        for i in self.data_label_distribution:
            self.data_label_distribution[i]= int(self.data_label_distribution[i] * self.val_ratio)

        self.immutable_data_label_distribution = copy.deepcopy(self.data_label_distribution)
        self.data_label_distribution = copy.deepcopy(self.immutable_data_label_distribution)

        #데이터 생성 진행
        if 'train' in self.mode:
            for i in range(self.seed):
                self.data_label_distribution = copy.deepcopy(self.immutable_data_label_distribution)
                for gen_age in self.data_label_paths:
                    paths = self.data_label_paths[gen_age]
                    while paths:
                        if self.data_label_distribution[gen_age] !=0:
                            self.data_label_distribution[gen_age] -= 1
                            profile = paths.pop()
                            img_folder = os.path.join(self.data_dir, profile)
                            for file_name in os.listdir(img_folder):
                                _file_name, ext = os.path.splitext(file_name)
                                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                                    continue
                                img_path = os.path.join(self.data_dir, profile,
                                                        file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                                mask_label = self._file_names[_file_name]
                                id, gender, race, age = profile.split("_")
                                gender_label = GenderLabels.from_str(gender)
                                age_label = AgeLabels.from_number(age)
                                self.image_paths.append(img_path)
                                self.mask_labels.append(mask_label)
                                self.gender_labels.append(gender_label)
                                self.age_labels.append(age_label)
                        elif self.data_label_distribution[gen_age] ==0:
                            break

            self.data_label_distribution = copy.deepcopy(self.immutable_data_label_distribution)
            for gen_age in self.data_label_paths:
                paths = self.data_label_paths[gen_age]
                while paths:
                    if self.data_label_distribution[gen_age] != 0:
                        self.data_label_distribution[gen_age] -= 1
                        paths.pop()
                        continue
                    elif self.data_label_distribution[gen_age] ==0:
                        profile = paths.pop()
                        img_folder = os.path.join(self.data_dir, profile)
                        for file_name in os.listdir(img_folder):
                            _file_name, ext = os.path.splitext(file_name)
                            if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                                continue
                            img_path = os.path.join(self.data_dir, profile,
                                                    file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                            mask_label = self._file_names[_file_name]
                            id, gender, race, age = profile.split("_")
                            gender_label = GenderLabels.from_str(gender)
                            age_label = AgeLabels.from_number(age)
                            self.image_paths.append(img_path)
                            self.mask_labels.append(mask_label)
                            self.gender_labels.append(gender_label)
                            self.age_labels.append(age_label)

        elif 'val' in self.mode:
            for i in range(self.seed):
                self.data_label_distribution = copy.deepcopy(self.immutable_data_label_distribution)
                for gen_age in self.data_label_paths:
                    paths = self.data_label_paths[gen_age]
                    while paths:
                        if self.data_label_distribution[gen_age] !=0:
                            self.data_label_distribution[gen_age] -= 1
                            paths.pop()
                        elif self.data_label_distribution[gen_age] ==0:
                            break

            self.data_label_distribution = copy.deepcopy(self.immutable_data_label_distribution)
            for gen_age in self.data_label_paths:
                paths = self.data_label_paths[gen_age]
                while paths:
                    if self.data_label_distribution[gen_age] != 0:
                        self.data_label_distribution[gen_age] -= 1
                        profile = paths.pop()
                        img_folder = os.path.join(self.data_dir, profile)
                        for file_name in os.listdir(img_folder):
                            _file_name, ext = os.path.splitext(file_name)
                            if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                                continue
                            img_path = os.path.join(self.data_dir, profile,
                                                    file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                            mask_label = self._file_names[_file_name]
                            id, gender, race, age = profile.split("_")
                            gender_label = GenderLabels.from_str(gender)
                            age_label = AgeLabels.from_number(age)
                            self.image_paths.append(img_path)
                            self.mask_labels.append(mask_label)
                            self.gender_labels.append(gender_label)
                            self.age_labels.append(age_label)
                        continue
                    elif self.data_label_distribution[gen_age] == 0:
                        break
                    
                    
        print(self.mode, self.image_paths[0])
        print(self.mode, self.image_paths[-1])


    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image#.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp= np.array(img_cp)
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        print(train_set)
        return train_set, val_set



class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            # Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)