import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataset import FolderImageTxtDataset
from model import EffB0 
from torchvision import transforms
from tqdm import tqdm
import argparse
import sys
import logging
import sys

# 设置日志格式和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("train.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--food101_root', type=str, default="E:/mlwork3", help='数据集根目录，如E:/mlwork3')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型参数路径，可选')
    parser.add_argument('--data_augment', type=bool, default=True, help='数据增强')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2正则化率')
    parser.add_argument('--save_path', type=str, default='./', help='模型保存地址')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

args = parse_args()

if args.data_augment == False:
    augment_flag = ""
else:
    augment_flag ="augment"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载自定义划分的数据集
train_txt = os.path.join(args.food101_root, 'TrainAndValList', 'train.lst')
val_txt = os.path.join(args.food101_root, 'TrainAndValList', 'validation.lst')

train_dataset = FolderImageTxtDataset(root_dir=args.food101_root, txt_file=train_txt, transform=train_transform, image_subdir="low-resolution")
val_dataset = FolderImageTxtDataset(root_dir=args.food101_root, txt_file=val_txt, transform=val_transform, image_subdir="low-resolution")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.idx_to_class)
model = EffB0(num_classes=num_classes).to(device)

# 可选加载预训练权重
if args.model_path is not None and os.path.exists(args.model_path):
    model.load(args.model_path)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

num_epochs = args.num_epochs

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    val_correct = 0
    for images,labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs,feature = model.forward(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        train_correct += correct
    model.eval()
    with torch.no_grad():
        for images,labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs,feature = model.forward(images)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()
            val_correct += correct
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    train_total = len(train_dataset)
    val_total = len(val_dataset)
    train_acc = train_correct/train_total * 100
    val_acc = val_correct / val_total * 100
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    torch.save(model.state_dict(),os.path.join(args.save_path, f'epoch{epoch}_food101_model_{augment_flag}.pth'))

# 保存训练时的类别顺序到classes.txt
with open('classes.txt', 'w', encoding='utf-8') as f:
    for cls in train_dataset.idx_to_class:
        f.write(f"{cls}\n") 