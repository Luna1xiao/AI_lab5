# 多模态情感分析

这是一个结合了文本和图像的多模态情感分析模型

Setup

This implemetation is based on Python3.8 To run the code, you need the following dependencies:

- numpy==1.21.4
- matplotlib==3.5.0
- argparse==1.1
- sklearn==0.0
- pandas==1.4.3
- pillow==9.1.0
- transformers==4.20.1
- torch==1.10.0+cu113
- torchvision==0.11.1+cu113

You can simply run 

```python
pip install -r requirements.txt
```

## 仓库结构
We select some important files for detailed description.

```python
├─data
│  └─data
│      ├─data
│      ├─test_without_label.txt
│      └─train.txt
├─bert-base-uncased #手动拉取的bert-base-uncased的参数
├─main.py	# 主函数
├─process_data.py	# 数据预处理
├─multi_model.py	# 多模态模型
├─requirements.txt	# 项目配置文件
├─train_multimodel.py	# 训练多模态模型并测试
├─train_not_multimodel.py	# 训练文本与图像的消融实验
├─picture_model.py	# 图像模型
└─result.txt  # 结果
```

运行模型

1.运行多模态模型

```
python main.py --model multi_model
```

2.运行只有文本的消融实验

```
python main.py --model only_text
```

3.运行只有图片的消融实验

```
python main.py --model only_picture
```

