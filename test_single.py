import os
import torch
from torch.utils.data import DataLoader
from model import EffB0
from torchvision import transforms
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('--test_image_path', type=str, default=r"test.jpg", help='推理图片路径')
    parser.add_argument('--model_path', type=str, default=r"epoch4_food101_model_augment.pth", help='模型路径')
    parser.add_argument('--csv_path', type=str, default="/hy-tmp/zhangqi/", help='csv文件路径')
    return parser.parse_args()

args = parse_args()

transforms_ = transforms.Compose([
    transforms.ToTensor(),                # 转为Tensor并归一化到[0,1]
    transforms.Resize((224, 224)),          # 等比缩放至短边匹配img_size
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet均值
        std=[0.229, 0.224, 0.225]        # ImageNet方差
    )
])

# 加载测试数据集
# test_dataset = TestDataset(train_dir=args.train_data_path,root_dir=args.test_data_path, transform=transforms_)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffB0().to(device)
model.baseline_extractor.classifier = torch.nn.Linear(in_features=1280, out_features=130).to("cuda")
# 加载训练好的模型
model.load_state_dict(torch.load(args.model_path))
# 获取类别映射
train_dir = "E:/mlwork3/low-resolution"  # 你的训练集路径
classes = os.listdir(train_dir)
classes.sort()  # 建议排序
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

img_path = args.test_image_path
name = "测试图片"
image = Image.open(img_path).convert('RGB')  # 统一转为RGB
image = np.array(image)

image = transforms_(image)
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
image = image.to(device)
model.eval()
outputs, _ = model.forward(image)
_, predicted = torch.max(outputs.data, 1)

output = predicted.item()
print("预测结果为：", output, idx_to_class[output])


