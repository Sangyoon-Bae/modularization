import torch
import pandas as pd
import numpy as np
import os # operating system에 접근.
from torch.utils.data import Dataset # 데이터 원하는 대로 로드 하기 (모델에 넣기 전에)
from torchvision import transforms # data를 우리가 원하는 형식으로 바꾸는 거
from PIL import Image

class TrainDogCatDataset(Dataset):
    def __init__(self, **kwargs):
        ## kwargs는 나중에 main.py에서 설정해서 다 넣어주게 됩니다
        
        super().__init__() # Dataset이라는 부모 클래스를 상속한다.
        self.train_data_dir = kwargs.get('train_data_dir')
        # kwargs는 args로 만든 딕셔너리. 그렇다면 .get(key 이름)을 하면 key와 pair인 value가 나와요.
        # set as 'C:/Users/stell/tutoring/Sarah/modularization/dogs_and_cats/train'
        self.transform = kwargs.get('transform')
        
    def __len__(self):
        return len(os.listdir(self.train_data_dir))
    
    def __getitem__(self, idx):
        # idx를 이용해서 이미지, 라벨을 로드
        # 이미지를 transformation (텐서로 바꾸고, 이미지 크기 다 똑같이 맞춰주고, normalization 하고 등등..)
        ## 텐서로 바꾸는 이유 : 딥러닝 모델이 텐서 인풋을 받기 때문..
        ## normalization 이유 : 텐서 값의 범위가 너무 넓으면 딥러닝 모델이 길을 잘 못 찾기 때문 (딥러닝은 생각보다 바보..)
        image_name = os.listdir(self.train_data_dir)[idx]
        image_path = os.path.join(self.train_data_dir, image_name)
        image = Image.open(image_path)
        image = np.array(image)/255
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        label = image_name.split('.')[0]
        if label == 'dog':
            label = 1.0
        else:
            label = 0.0
        
        if self.transform:
            image = transform(image)
            image = image.type(torch.FloatTensor)
        
        return image, label

class TestDogCatDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__() # Dataset이라는 부모 클래스를 상속한다.
        self.test_data_dir = kwargs.get('test_data_dir')
        # set as 'C:/Users/stell/tutoring/Sarah/modularization/dogs_and_cats/test1'
        self.transform = kwargs.get('transform')
        self.label_csv = pd.read_csv('C:/Users/stell/tutoring/Sarah/modularization/dogs_and_cats/sampleSubmission.csv')
        
    def __len__(self):
        return len(os.listdir(self.test_data_dir))
    
    def __getitem__(self, idx):
        # idx를 이용해서 이미지, 라벨을 로드
        # 이미지를 transformation (텐서로 바꾸고, 이미지 크기 다 똑같이 맞춰주고, normalization 하고 등등..)
        ## 텐서로 바꾸는 이유 : 딥러닝 모델이 텐서 인풋을 받기 때문..
        ## normalization 이유 : 텐서 값의- 범위가 너무 넓으면 딥러닝 모델이 길을 잘 못 찾기 때문 (딥러닝은 생각보다 바보..)
        image_name = os.listdir(self.testdata_dir)[idx]
        image_path = os.path.join(self.test_data_dir, image_name)
        image = Image.open(image_path)
        image = np.array(image)/255
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        
        label = self.label_csv.iloc[idx, 1]
        if self.transform:
            image = transform(image)
            image = image.type(torch.FloatTensor)
        
        return image, label


# class FlowerDataset(Dataset):
#     def __init__(self, **kwargs):
#         super().__init__() # Dataset이라는 부모 클래스를 상속한다.
#         self.root_dir = kwargs.get('root_dir')
#         self.transform = kwargs.get('transform')
#         self.classes = os.listdir(root_dir)
        
#     def __len__(self):
#         return sum([len(files) for _, _, files in os.walk(self.root_dir)])
            
#     def __getitem__(self, idx):
#         # idx를 이용해서 이미지, 라벨을 로드
#         # 이미지를 transformation (텐서로 바꾸고, 이미지 크기 다 똑같이 맞춰주고, normalization 하고 등등..)
#         ## 텐서로 바꾸는 이유 : 딥러닝 모델이 텐서 인풋을 받기 때문..
#         ## normalization 이유 : 텐서 값의 범위가 너무 넓으면 딥러닝 모델이 길을 잘 못 찾기 때문 (딥러닝은 생각보다 바보..)
#         class_folder = self.classes[idx // len(self.classes)]
#         image_files = os.listdir(os.path.join(self.root_dir, class_folder))
#         img_name = os.path.join(self.root_dir, class_folder, image_files[idx % len(image_files)])
#         image = Image.open(img_name)
        
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((224,224)),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
#         ])
        
#         if self.transform:
#             image = transform(image)
        
#         return image, idx // len(self.classes)
        