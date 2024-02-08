import torch
import numpy as np
import os
import argparse
from trainer import Trainer

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='240207_cat_dog', help='experiment name')
    parser.add_argument('--train_data_dir', type=str, default = 'C:/Users/stell/tutoring/Sarah/modularization/dogs_and_cats/train')
    parser.add_argument('--test_data_dir', type=str, default = 'C:/Users/stell/tutoring/Sarah/modularization/dogs_and_cats/test1')
    parser.add_argument('--transform', action='store_true')
    
    args = parser.parse_args()
        
    return args

    
if __name__ == '__main__':
    args = get_arguments()
    # args는 Namespace라는 특이한 데이터 타입.
    kwargs = vars(args)
    # kwargs는 이제 딕셔너리가 됐어요!
    
    trainer = Trainer(**kwargs)
    trainer.training()