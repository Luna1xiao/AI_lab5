import torch
import torch.nn as nn
from transformers import BertModel
from picture_model import *

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        if x.dim() != 3:
            x = x.view(x.size(0), -1, 4096)  
        attn_weights = torch.softmax(self.attn(x), dim=1)
        output = torch.bmm(attn_weights.permute(0, 2, 1), x)
        return output.squeeze(1)



class MutilModalClassifier(nn.Module):
    def __init__(self, model_type='multi_model', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(MutilModalClassifier, self).__init__()
        self.model_type = model_type
        self.device = device
        self.bert = BertModel.from_pretrained("./bert-base-uncased")
        self.vgg = VGG16()
        self.attention_bert = Attention(hidden_dim=768)
        self.attention_vgg = Attention(hidden_dim=4096)  # 注意力机制的输入维度根据VGG16的输出维度而定
        self.fc = nn.Sequential(
            nn.Linear(768 + 4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3),
        )
        for param in self.bert.parameters():
            param.requires_grad_(False)

    def forward(self, pic, input_ids, token_type_ids, attention_mask):
        if self.model_type == 'multi_model':
            bert_out = self.bert(input_ids, token_type_ids, attention_mask)
            #print("BERT output shape:", bert_out.last_hidden_state.shape)

            bert_attn_output = self.attention_bert(bert_out.last_hidden_state)
            #print("BERT output shape:", bert_out.last_hidden_state.shape)
            #print("VGG output shape:", vgg_feature.shape)

            vgg_feature = self.vgg(pic)
            #print("VGG output shape:", vgg_feature.shape)
            vgg_attn_output = self.attention_vgg(vgg_feature)
            return self.fc(torch.cat((bert_attn_output, vgg_attn_output), dim=1))
        elif self.model_type == 'only_picture':
            vgg_feature = self.vgg(pic)
            vgg_attn_output = self.attention_vgg(vgg_feature)  
            return self.fc(torch.cat((vgg_attn_output, torch.zeros(size=(vgg_feature.size(0), 768)).to(self.device)), dim=1))
        elif self.model_type == 'only_text':
            bert_out = self.bert(input_ids, token_type_ids, attention_mask)
            bert_attn_output = self.attention_bert(bert_out.last_hidden_state) 
            return self.fc(torch.cat((bert_attn_output, torch.zeros(size=(bert_out.last_hidden_state.size(0), 4096)).to(self.device)), dim=1))

    
