import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from process_data import *
import torch.optim as optim
import numpy as np
import argparse

from train_multimodel import *
from train_not_multimodel import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi arg")
    parser.add_argument('--model', default='multi_model')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.90, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    model_type = args.model
    LR = args.lr
    momentum = args.momentum
    batch_size = args.batch_size
    train_dataloader, valid_dataloader, test_dataloader = prepare_data_loaders(batch_size)   
    if model_type == 'multi_model':
        run_multi_model(LR, momentum, train_dataloader, valid_dataloader, test_dataloader)       
    elif model_type == 'only_picture':
        run_model(LR, momentum, train_dataloader, valid_dataloader, test_dataloader,'only_picture')     
    elif model_type == 'only_text':
        run_model(LR, momentum, train_dataloader, valid_dataloader, test_dataloader,'only_text')