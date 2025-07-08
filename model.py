import os
import sys
import copy
import weakref
import contextlib
import numpy as np
from typing import Iterable, Optional
from PIL import Image
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Dropout, Module
from torchvision import transforms

class EffB0(nn.Module):
    def __init__(self,  num_classes=130, pretrained=True):
        super().__init__()
        self.name = 'EffB0'
        self.baseline_extractor = timm.create_model('tf_efficientnet_b0.ns_jft_in1k', pretrained=False, num_classes=num_classes)
        
        
    def forward(self, x):
        x = self.baseline_extractor.forward_features(x)
        x = self.baseline_extractor.global_pool(x)
        if self.baseline_extractor.drop_rate > 0.:
            x = F.dropout(x, p=self.baseline_extractor.drop_rate, training=self.baseline_extractor.training)
        feature = x
        x = self.baseline_extractor.classifier(x)
        # x  = F.softmax(x, dim=1)
        return x, feature
    


    def load(self, path):
        weights_file = torch.load(path, map_location='cpu')
        # 过滤掉分类器层的参数
        filtered_weights = {k: v for k, v in weights_file.items() if "classifier" not in k}
        self.baseline_extractor.load_state_dict(filtered_weights, strict=False)
        # 重新初始化分类器
        self.baseline_extractor.classifier = nn.Linear(in_features=1280, out_features=130).to(next(self.parameters()).device)
        print("预训练模型加载成功！")
        # weights_file = torch.load(path)
        # new_state_dict = {}
        # for key, value in weights_file.items():
        #     new_key = 'baseline_extractor.' + key  # 在原有的键前添加前缀
        #     new_state_dict[new_key] = value
        # self.load_state_dict(new_state_dict, strict=False)
        # self.baseline_extractor.classifier = nn.Linear(in_features=1280, out_features=36).to("cuda")#输出类别数量

        
# #测试函数
if __name__ == '__main__':
    # 创建 EffB0 实例
    model = EffB0()
    model.load(r"D:\桌面文件\大三下\机器学习课程设计\task3\model\pytorch_model.bin")
    image_tensor = torch.randn(1, 3, 224, 224)
    # 确保你的环境有可用的 GPU，否则注释掉 .cuda()
    with torch.no_grad():  # 在推理模式下关闭梯度计算
        pred = model.forward(image_tensor)
        print(f"预测结果: {pred}")