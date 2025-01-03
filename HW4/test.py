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
from resnet import *
from torch.autograd import Variable
import torch
import torch.nn as nn


class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.data_dir = os.path.join(args.raf_path, 'Images', phase)
        self.aug_func = [flip_image, add_g]

        # Load label mapping
        self.file_paths = []
        self.labels = []
        if phase == 'test':
            # For test phase, load files from the test directory
            image_files = [
                f for f in os.listdir(args.test_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            for image_file in image_files:
                file_path = os.path.join(args.test_dir, image_file)
                self.file_paths.append(file_path)
                self.labels.append(-1)

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
        if self.transform:
            image = self.transform(image)
        return image, label, idx


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class res18feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0.4, out_dim=512):
        super(res18feature, self).__init__()
        res18 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000)

        if pretrained:
            pt_model = torch.load('./model/ResNet18.pth')
            state_dict = pt_model['state_dict']
            res18.load_state_dict(state_dict, strict=False)

        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = F.normalize(x, p=2, dim=1)
        return x
    
class res50feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(res50feature, self).__init__()
        res50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
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
        output = F.normalize(output, p=2, dim=1)
        return output


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded successfully!")
    return model


def load_ensemble_models(args):
    model_paths = ['./model/v0.pth', './model/v1.pth']
    models = []

    for i, model_path in enumerate(model_paths):
        # if i < len(model_paths) - 1:
        #     model = res18feature(pretrained=True, num_classes=7)
        # else:
        #     model = res50feature(pretrained=True, num_classes=7)
        model = res18feature(pretrained=True, num_classes=7)

        checkpoint = torch.load(model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(args.device)
        model.eval()
        models.append(model)

    print("All ensemble models loaded successfully.")
    return models


def test_ensemble(args):
    # Define transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load test dataset
    test_dataset = RafDataset(args, phase='test', transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load ensemble models
    ensemble_models = load_ensemble_models(args)

    # Inference
    predictions = []
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing Ensemble", leave=False)
        for imgs, _, file_indices in test_progress:
            imgs = imgs.to(args.device)

            # Collect predictions from each model
            ensemble_outputs = []
            for model in ensemble_models:
                outputs = model(imgs)
                ensemble_outputs.append(outputs)

            # Combine predictions (e.g., averaging softmax probabilities)
            ensemble_outputs = torch.stack(ensemble_outputs, dim=0)  # Shape: (num_models, batch_size, num_classes)
            avg_outputs = torch.mean(ensemble_outputs, dim=0)  # Shape: (batch_size, num_classes)
            _, predicted_labels = torch.max(avg_outputs, 1)  # Take the argmax of averaged outputs

            for idx, label in zip(file_indices, predicted_labels.cpu().numpy()):
                filename_without_extension = os.path.splitext(os.path.basename(test_dataset.file_paths[idx]))[0]
                predictions.append((filename_without_extension, label))

    # Save predictions
    with open(args.output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])
        writer.writerows(predictions)
    print(f"Ensemble predictions saved to {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./data', help='Path to RAF dataset')
    parser.add_argument('--test_dir', type=str, default='./data/Images/test', help='Directory containing test images')
    parser.add_argument('--output_csv', type=str, default='./output.csv', help='Path to save the ensemble output CSV')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference (cpu or cuda)')
    args = parser.parse_args()

    test_ensemble(args)

