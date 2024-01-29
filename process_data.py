import os
import io
import numpy as np
import torch
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def prepare_data_loaders(batch_size):
    # 初始化BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    
    # 定义数据路径
    data_directory = './data/data/data'
    train_file_path = './data/data/train.txt'
    test_file_path = './data/data/test_without_label.txt'

    # 读取训练数据和测试数据
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    # 将标签转换为数字
    def convert_tags_to_ids(data):
        length = len(data)
        for i in range(length):
            if data['tag'][i] == 'negative':
                data['tag'][i] = 0
            elif data['tag'][i] == 'positive':
                data['tag'][i] = 1
            elif data['tag'][i] == 'neutral':
                data['tag'][i] = 2
        return data

    train_data = convert_tags_to_ids(train_data)

    # 获取并处理数据
    def process_data(data, tokenizer):
        pic_data = []
        text_data = []
        for item in data['guid']:
            pic_path = os.path.join(data_directory, str(item) + ".jpg")
            text_path = os.path.join(data_directory, str(item) + ".txt")
            img = Image.open(pic_path)
            text = open(text_path, 'r', encoding='utf-8', errors='ignore').read()
            text2id = tokenizer(text, max_length=128, padding="max_length", truncation=True)

            pic_data.append(np.asarray(img.resize((224, 224)), dtype=np.float32).transpose((2, 0, 1)))
            text_data.append(text2id)

        data['pic'] = pic_data
        data['text'] = text_data
        return data

    train_data = process_data(train_data, bert_tokenizer)
    test_data = process_data(test_data, bert_tokenizer)

    # 划分训练集和验证集
    train_data, valid_data = train_test_split(train_data, test_size=0.2)

    train_data.reset_index(inplace=True)
    valid_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)

    # 定义数据集类
    class MultiModalDataset(Dataset):
        def __init__(self, data, is_test=False):
            self.data = data
            self.is_test = is_test

        def __getitem__(self, index):
            item = (
                self.data.loc[index]['pic'],
                torch.tensor(self.data.loc[index]['text']['input_ids']),
                torch.tensor(self.data.loc[index]['text']['token_type_ids']),
                torch.tensor(self.data.loc[index]['text']['attention_mask'])
            )
            if not self.is_test:
                return (*item, self.data.loc[index]['tag'])
            else:
                return (*item, self.data.loc[index]['guid'])

        def __len__(self):
            return len(self.data)

    # 创建数据加载器
    train_dataset = MultiModalDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = MultiModalDataset(valid_data)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MultiModalDataset(test_data, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
