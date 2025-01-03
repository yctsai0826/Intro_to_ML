# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F

from utils import *
from torch.autograd import Variable

open_num = 0

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None, val_split=0.05):
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.data_dir = os.path.join(args.raf_path, 'Images', 'train')
        self.aug_func = [flip_image, add_g]
        class_mapping = self._load_class_mapping(args.label_path)

        self.file_paths = []
        self.labels = []

        # 處理訓練和驗證集的劃分
        image_files = []
        labels = []
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(class_path, img_file))
                        labels.append(class_mapping.get(class_name, -1))

        # 確保標籤範圍正確
        assert all(0 <= label < 7 for label in labels), "Labels out of range!"

        # 隨機劃分訓練和驗證集
        data = list(zip(image_files, labels))
        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_split))
        if phase == 'train':
            data = data[:split_idx]
        elif phase == 'val':
            data = data[split_idx:]
        else:
            raise ValueError("Invalid phase. Use 'train' or 'val'.")

        self.file_paths, self.labels = zip(*data)
        print(f"Loaded {len(self.file_paths)} samples for phase '{phase}'")

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        if image is None:
            raise ValueError(f"Failed to load image at {file_path}")

        image = image[:, :, ::-1]  # BGR -> RGB
        if self.phase == 'train' and self.basic_aug:
            if random.uniform(0, 1) > 0.5:
                index = random.randint(0, len(self.aug_func) - 1)
                image = self.aug_func[index](image)  # 使用增強函數

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


    def _load_class_mapping(self, label_path):
        with open(label_path, 'r') as f:
            mapping = json.load(f)
        return mapping

def add_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def rotate_image(image, angle_range=(-30, 30)):
    height, width = image.shape[:2]
    angle = random.uniform(*angle_range)  # 隨機選擇一個角度
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class res50feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(res50feature, self).__init__()
        res50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(res50.children())[:-2])  # 截取 backbone

        # 計算輸出特徵的大小
        example_input = torch.randn(1, 3, 224, 224)  # 假設輸入大小為 224x224
        example_output = self.features(example_input)
        flattened_size = example_output.view(example_output.size(0), -1).shape[1]

        # 動態設置全連接層
        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x, target=None, phase='train'):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='./data', help='raf_dataset_path')
parser.add_argument('--label_path', type=str, default='./data/index_mapping', help='label_path')
parser.add_argument('--workers', type=int, default=2, help='number of workers')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--out_dimension', type=int, default=2048, help='feature dimension')
args = parser.parse_args()


def train():
    setup_seed(0)
    res50 = res50feature()
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # 確保輸入大小為 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RafDataset(args, phase='train', transform=data_transforms)
    val_dataset = RafDataset(args, phase='val', transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    res50 = res50.cuda()
    res50 = torch.nn.DataParallel(res50)
    params = res50.parameters()

    optimizer = torch.optim.Adam([{'params': params}], lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0
    best_epoch = 0
    for i in range(1, args.epochs + 1):
        print(f'Epoch {i}/{args.epochs}')

        # Training Loop
        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        res50.train()
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for imgs, labels, indexes in train_progress:
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = res50(imgs, labels, phase='train')
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss.item()

            train_progress.set_postfix(loss=loss.item())

        scheduler.step()
        running_loss = running_loss / iter_cnt
        train_acc = correct_sum.float() / float(train_dataset.__len__())
        print(f"Train Acc: {train_acc:.4f}, Train Loss: {running_loss:.4f}")

        # Validation Loop
        val_loss = 0.0
        val_correct_sum = 0
        res50.eval()
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc="Validating", leave=False):
                imgs = imgs.cuda()
                labels = labels.cuda()

                outputs = res50(imgs, labels, phase='val')
                loss = criterion(outputs, labels)

                _, predicts = torch.max(outputs, 1)
                val_correct_num = torch.eq(predicts, labels).sum()
                val_correct_sum += val_correct_num
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc = val_correct_sum.float() / float(val_dataset.__len__())
        print(f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = i
        torch.save({'model_state_dict': res50.module.state_dict()}, f"./model/epoch_{i}.pth")

    print(f'Best Val Acc: {best_acc:.4f}, Best Epoch: {best_epoch}')
    
if __name__ == '__main__':
    train()
