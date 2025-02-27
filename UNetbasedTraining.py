#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:20:05 2025

@author: ssl
"""



import numpy as np
from scipy.signal import chirp
import torch.optim as optim
import matplotlib.pyplot as plt
from stockwell import st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
import random
from torch.nn.utils import weight_norm
import logging

torch.cuda.empty_cache()  # GPU 캐시 비우기


def setup_logging(log_file="training.log"):
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 새 핸들러 추가
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
# 로깅 설정 적용
setup_logging("training.log")
logging.info("Logging setup complete!")

# Random seed 고정
def set_seed(seed):
    torch.manual_seed(seed)  # CPU 연산 시드 고정
    torch.cuda.manual_seed(seed)  # GPU 연산 시드 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 동일한 시드 적용
    np.random.seed(seed)  # NumPy 시드 고정
    random.seed(seed)  # Python 랜덤 시드 고정
    torch.backends.cudnn.deterministic = True  # 연산의 결정론적 결과 보장
    torch.backends.cudnn.benchmark = False  # 특정 환경에서 속도 저하 가능성 있음

# 시드 설정
set_seed(5148)

class UNetQRS(nn.Module):
    def __init__(self, num_classes):
        super(UNetQRS, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self._conv_block(2, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder (Upsampling)
        self.dec4 = self._upconv_block(1024 + 512, 512)
        self.dec3 = self._upconv_block(512 + 256, 256)
        self.dec2 = self._upconv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)

        # Final Convolution (Output layer)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_channels, out_channels):
        """Upsampling + Convolution block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder
        b_upsampled = nn.functional.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([b_upsampled, e4], dim=1))
        d4_upsampled = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d4_upsampled, e3], dim=1))
        d3_upsampled = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d3_upsampled, e2], dim=1))
        d2_upsampled = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d2_upsampled, e1], dim=1))

        # Final output logits
        out = self.final_conv(d1)  # Shape: (batch_size, num_classes, 151, 1000)
        out = out.mean(dim=2)      # Collapse height dimension -> Shape: (batch_size, num_classes, 1000)
        return out  # Logits


    
class CustomECGDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            signal_tensor = torch.tensor(np.load(self.file_paths[idx]), dtype=torch.float32)
            label = torch.tensor(np.loadtxt(self.labels[idx], delimiter=","),dtype=torch.long)
        except Exception as e:
            print(f"Error at index {idx}: {str(e)}. Returning zero tensor.")
            signal_tensor = torch.zeros((2, 151, 1000), dtype=torch.float32)
            label = torch.zeros((1,1000),dtype=torch.long)
        return signal_tensor, label
    
def calculate_iou(predictions, labels, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        # 클래스별 마스크 생성
        pred_mask = (predictions == cls)
        label_mask = (labels == cls)
        
        # 교집합과 합집합 계산
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        
        if union == 0:
            iou = float('nan')  # 해당 클래스가 없을 경우 NaN 처리
        else:
            iou = intersection / union
        iou_per_class.append(iou)
    return iou_per_class

#%% Main Code
DATA_NAME="QTDB"
num_classes=4
batch_size=8
num_workers=0
patience = 8
num_epochs = 50

file_paths = []
labels = []
        
file_folder_name="./"+DATA_NAME+"_STransformed_npy/"
label_folder_name="./"+DATA_NAME+"_segment_label/"

for data_name in os.listdir(file_folder_name):
    file_path=file_folder_name+data_name # data_name="1.npy"
    file_paths.append(file_path)
    label_name = data_name.replace(".npy","_label.csv") # label_name = "1_plabel.csv"
    labels.append(label_folder_name+label_name)
    
# 1차 분할: train (80%), valid+test (20%)
train_data, valid_test_data, train_labels, valid_test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, shuffle=True
)

# 2차 분할: valid (50% of 20% = 10%), test (50% of 20% = 10%)
valid_data, test_data, valid_labels, test_labels = train_test_split(
    valid_test_data, valid_test_labels, test_size=0.5, random_state=42
)

# TensorDataset으로 변환
train_dataset = CustomECGDataset(train_data, train_labels)
valid_dataset = CustomECGDataset(valid_data, valid_labels)
test_dataset = CustomECGDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ratio_0 = 0.9
# ratio_1 = 0.1

# # 양성 클래스(1)에 대한 가중치
# pos_weight = torch.tensor([ratio_0 / ratio_1])  # 9.0

# BCEWithLogitsLoss 정의
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
criterion = nn.CrossEntropyLoss()

model = UNetQRS(num_classes=num_classes)
model.to(device)
# predictions = model(input_data)  # (batch, 1000)
# loss = criterion(predictions, target_labels)  # target_labels: (batch, 1000)

# Optimizer와 Scheduler 설정
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 학습률 감소

# 학습, 검증, 테스트 결과 저장용 리스트
train_losses = []
valid_losses = []
best_valid_loss = float("inf")  # Validation Loss 최저값 추적

# 학습 루프
patient = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for signals, labels in train_loader:
        # 데이터 로드 및 디바이스 이동
        signals, labels = signals.to(device), labels.to(device)

        # 모델 예측 및 손실 계산
        optimizer.zero_grad()
        outputs = model(signals)  # (batch, 1000)
        loss = criterion(outputs, labels)  # (batch, 1000)

        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        # 배치 손실 합산
        train_loss += loss.item()

    # 학습 손실 저장
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 검증 루프
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for signals, labels in valid_loader:
            signals, labels = signals.to(device), labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)

    # Validation Loss가 개선되면 모델 저장
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "./best_binary_segmentation_model.pth")
        logging.info(f"Model saved at epoch {epoch+1} with validation loss: {valid_loss:.4f}")
        patient=0
    else:
        patient+=1
    # 로그 출력
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    
    # Scheduler 업데이트
    scheduler.step()
    if patient == patience:
        break

# 학습 후 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

model.eval()
test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    total_iou = [0] * 4  # 4개의 클래스에 대한 IoU 합산
    total_count = [0] * 4  # 클래스별 데이터 개수

    for signals, label in test_loader:
        signals, label = signals.to(device), label.to(device)
        
        outputs = model(signals)
        predictions = torch.argmax(outputs, dim=1)  # Shape: (batch_size, sequence_length)
        loss = criterion(outputs, label)
        test_loss += loss.item()
        
        # IoU 계산
        batch_iou = calculate_iou(predictions.cpu(), label.cpu(), num_classes=4)
        
        # IoU 합산 (NaN 값은 제외)
        for cls in range(4):
            if not torch.isnan(torch.tensor(batch_iou[cls])):
                total_iou[cls] += batch_iou[cls]
                total_count[cls] += 1
        
        # 예측값과 실제 라벨 저장
        all_preds.append(predictions.cpu().numpy())
        all_labels.append(label.cpu().numpy())

# 평균 IoU 계산
mean_iou = [total_iou[cls] / total_count[cls] if total_count[cls] > 0 else 0 for cls in range(4)]
logging.info(f"Mean IoU per class: {mean_iou}")
test_loss /= len(test_loader)
logging.info(f"Test Loss: {test_loss:.4f}")

test_number = []

for i in range(len(test_data)):
    test_name = test_data[i]
    number = ""
    for spell in test_name:
        if spell in "1234567890":
            number += spell
    
    # 테스트 데이터 로드
    test_pdata = np.loadtxt(f"./{DATA_NAME}_preprocessed_signal/{number}.csv", delimiter=",")
    data_pred = all_preds[i // batch_size][(i) % batch_size]  # 배치에서 해당 데이터의 예측값
    data_label = all_labels[i // batch_size][(i) % batch_size]
    
    # 신호 플롯
    for t in range(100, 900):
        if data_pred[t] == 1: #P
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='red')
        elif data_pred[t] == 2: #QRS
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='green')
        elif data_pred[t] == 3: # T wave
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='orange')
        else: #Normal
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='blue')
    
    plt.title(f"Prediction Highlights. {number} ")
    plt.xlabel("Time Index")
    plt.ylabel("Signal Value")
    plt.show()
    
    for t in range(100, 900):
        if data_label[t] == 1: #P
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='red')
        elif data_label[t] == 2: #QRS
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='green')
        elif data_label[t] == 3: # T wave
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='orange')
        else: #Normal
            plt.plot([t, t + 1], [test_pdata[t], test_pdata[t + 1]], color='blue')
    
    plt.title("Correct Segment")
    plt.xlabel("Time Index")
    plt.ylabel("Signal Value")
    plt.show()
    

    
