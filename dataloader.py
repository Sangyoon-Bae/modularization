import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler
from datasets import TrainDogCatDataset, TestDogCatDataset
import os

def get_datasets(**kwargs):
    batch_size = 32
    
    train_dataset = TrainDogCatDataset(**kwargs, test=False)
    valid_dataset = TrainDogCatDataset(**kwargs, test=False)
    test_dataset = TestDogCatDataset(**kwargs, test=True)
    
    print('loading splits')
    
    split_path = 'dog_cat_splits.txt'
    subject_list = open(split_path, 'r').readlines()
    train_index = np.argmax(['train' in line for line in subject_list])
    valid_index = np.argmax(['val' in line for line in subject_list])
    train_names = subject_list[train_index+1:valid_index]
    valid_names = subject_list[valid_index+1:]
    
    train_names = [names.split('\n')[0] for names in train_names]
    valid_names = [names.split('\n')[0] for names in valid_names]
    #subj_idx = os.listdir('C:/Users/stell/tutoring/Sarah/modularization/dogs_and_cats/train') # 나중에 kwargs로 대체
    subj_idx = os.listidr(kwargs.get('train_data_dir'))
    train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
    valid_idx = np.where(np.in1d(subj_idx, valid_names))[0].tolist()
    
    
    
    ## subset으로 우리가 만든 splits.txt를 기준으로 train, valid, test dataset을 나눔
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)
    
    # dataloader에 얹어서 모델이 데이터 뭉탱이 안에서 데이터를 차례대로 꺼내서 학습할 수 있도록!!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, valid_loader, test_loader