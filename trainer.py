import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import BinaryAccuracy
import numpy as np
import os
from tqdm import tqdm
from dataloader import get_datasets
from model import CNN

class Trainer():
    def __init__(self, **kwargs):
        # args 설정
        self.exp_name = kwargs.get('exp_name')
        
        # 필요한 것
        ## 1. model
        self.model = CNN()
        
        ## 2. optimizer
        self.create_optimizer()
        
        ## 3. loss function and metric
        self.criterion = nn.CrossEntropyLoss() # 온갖 종류의 classification에서 사용 가능
        self.acc = BinaryAccuracy()
        
        ## 4. data가 필요한데 이건 dataloader에서 가지고 올거예요.
        self.train_loader, self.valid_loader, self.test_loader = get_datasets(**kwargs)
        
        ## 5. epoch 개수 지정
        self.num_epochs = 30
        
        
        
    def create_optimizer(self):
        params = self.model.parameters()
        self.optimizer = optim.RMSprop(params)
        
    def train_epoch(self):
        # 1. 모델을 train 모드로 바꾸기
        self.model.train()
        # 2. loss, acc 초기화
        train_loss = 0
        train_acc = 0
        # 3. train_loader에서 데이터 꺼내서 돌리기
        for images, labels in tqdm(self.train_loader, position=0, leave=True):
            ## 3-1. optimizer 초기화
            self.optimizer.zero_grad()
            
            ## 3-2. loss 계산 및 backward
            outputs = self.model(images)
            outputs = outputs.squeeze() # https://pytorch.org/docs/stable/generated/torch.squeeze.html
            # 개수가 1인 차원을 없애줌!
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            ## 3-3. optimizer 업데이트
            self.optimizer.step()
            
            ## 3-4. acc 업데이트
            self.acc.update(outputs, labels)
            
            ## 3-5. train_loss, train_acc 업데이트
            train_loss += loss.item()
            train_acc += self.acc.compute()
        
        train_loss /= len(self.train_loader)
        train_acc /= len(self.train_loader)
        
        self.acc.reset()
        
        return train_loss, train_acc
            
        
    def valid_epoch(self):
        # 1. 모델을 eval 모드로 변경 (backpropagation을 막음 = 시험 보는 중에 공부 하지 마)
        self.model.eval()
        
        # 2. loss, acc 초기화
        valid_loss = 0
        valid_acc = 0
        
        # 3. 미분 막기
        with torch.no_grad():
            # 4. valid_loader에서 데이터 꺼내서 돌리기
            for images, labels in tqdm(self.valid_loader, position=0, leave=True):
                ## 4-1. loss 계산
                outputs = self.model(images)
                outputs = outputs.squeeze() # https://pytorch.org/docs/stable/generated/torch.squeeze.html
                # 개수가 1인 차원을 없애줌!
                loss = self.criterion(outputs, labels)
                
                ## 4-2. acc 업데이트
                self.acc.update(outputs, labels)

                ## 4-3. valid_loss, valid_acc 업데이트
                valid_loss += loss.item()
                valid_acc += self.acc.compute()

        valid_loss /= len(self.valid_loader)
        valid_acc /= len(self.valid_loader)

        self.acc.reset()
        
        return valid_loss, valid_acc
        
    def training(self):
        max_val_acc = 0.0
        # 모든 epoch에 대해서 반복!
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.valid_epoch()
            print(f'In epoch {epoch}, train loss is {train_loss}, train acc is {train_acc}, valid loss is {valid_loss}, valid acc is {valid_acc}.')
            
            if max_val_acc < valid_acc:
                max_val_acc = valid_acc
                print(f'saving best model with accuracy {max_val_acc}')
                torch.save(self.model.state_dict(), f"./experiments/{self.exp_name}/epoch_{epoch}_valid_acc_{max_val_acc}.pth")