import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class FolderImageTxtDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None, image_subdir="low-resolution"):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = []
        self.image_subdir = image_subdir

        # 读取所有图片路径
        with open(txt_file, 'r', encoding='utf-8-sig') as f:
            img_paths = [line.strip().replace('.//', '') for line in f.readlines() if line.strip()]

        # 提取所有类别名
        class_names = sorted(list(set([p.split('/')[0] for p in img_paths])))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.idx_to_class = class_names

        # 构建样本列表
        for rel_path in img_paths:
            class_name = rel_path.split('/')[0]
            label = self.class_to_idx[class_name]
            img_path = os.path.join(root_dir, self.image_subdir, rel_path)
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.img_names = []
        for file in os.listdir(self.root_dir):
            if self._is_valid_file(file):
                image_path = os.path.join(self.root_dir, file)
                self.img_paths.append(image_path)
                self.img_names.append(file)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = self.img_names[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, name
    def _is_valid_file(self, filename):
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(filename.lower().endswith(ext) for ext in valid_ext) 