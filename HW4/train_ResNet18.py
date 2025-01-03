# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
import matplotlib.pyplot as plt


# Dataset 定義
class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None, val_split=0.05):
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.data_dir = os.path.join(args.raf_path, 'Images', 'train')
        self.aug_func = [self.flip_image, self.add_g, self.rotate_image]
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

        return image, label

    def _load_class_mapping(self, label_path):
        with open(label_path, 'r') as f:
            mapping = json.load(f)
        return mapping

    @staticmethod
    def flip_image(image):
        return cv2.flip(image, 1)

    @staticmethod
    def add_g(image):
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    @staticmethod
    def rotate_image(image, angle_range=(-30, 30)):
        height, width = image.shape[:2]
        angle = random.uniform(*angle_range)
        
        # 计算旋转后图像的边界大小
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # 调整旋转矩阵以考虑新的边界
        rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
        rotation_matrix[1, 2] += (new_height / 2) - (height / 2)
        
        # 进行旋转，设置边界填充方式为镜像
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_REFLECT
        )
        
        return rotated_image

# 模型定義
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18Model, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 訓練函數
def train(args):
    # 變數初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Model(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    # 資料增強與載入
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RafDataset(args, phase='train', transform=data_transforms)
    val_dataset = RafDataset(args, phase='val', transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 用于绘制曲线的数据
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
        torch.save(model.state_dict(), f"./weights/epoch_{epoch+1}.pth")

        scheduler.step()

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")

    # 绘制学习曲线
    plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./data', help='Path to RAF dataset')
    parser.add_argument('--label_path', type=str, default='./data/index_mapping', help='Path to label mapping')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    args = parser.parse_args()

    train(args)
