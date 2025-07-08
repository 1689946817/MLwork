import os
import torch
from torch.utils.data import DataLoader
from dataset import TestDataset
from model import EffB0
from torchvision import transforms
import argparse
import pandas as pd
from tqdm import tqdm
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('--train_data_path', type=str, default="/hy-tmp/zhangqi/frutable/train/", help='训练数据路径')
    parser.add_argument('--test_data_path', type=str, default="/hy-tmp/zhangqi/frutable/test/", help='测试数据路径')
    parser.add_argument('--model_path', type=str, default="/hy-tmp/zhangqi/epoch99_model_augment.pth", help='模型路径')
    parser.add_argument('--csv_path', type=str, default="/hy-tmp/zhangqi/", help='csv文件路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    return parser.parse_args()

args = parse_args()

transforms_ = transforms.Compose([
    transforms.Resize((224, 224)),          # 等比缩放至短边匹配img_size
    transforms.ToTensor(),                # 转为Tensor并归一化到[0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet均值
        std=[0.229, 0.224, 0.225]        # ImageNet方差
    )
])

# 加载测试数据集
test_dataset = TestDataset(train_dir=args.train_data_path,root_dir=args.test_data_path, transform=transforms_)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffB0().to(device)
model.baseline_extractor.classifier = torch.nn.Linear(in_features=1280, out_features=130).to("cuda")
# 加载训练好的模型
model.load_state_dict(torch.load(args.model_path))



image_name=[]
image_label=[]

with torch.no_grad():

    for images, names in tqdm(test_loader):
        images = images.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(names)):
            image_name.append(names[i])
            image_label.append(predicted[i].item())

prediction = pd.DataFrame()
prediction["file"] = image_name
prediction["species"] = image_label
prediction["species"] = prediction["species"].astype(int).map(test_dataset.idx_to_class)
prediction.to_csv(args.csv_path+"eff_pre.csv",index=False)
