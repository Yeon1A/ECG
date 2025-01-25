#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:51:46 2025

@author: ssl
"""

#%% library (useless in)
import wfdb
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample
import torch
import numpy as np
from scipy.sparse import spdiags
import neurokit2 as nk
from multiprocessing import Pool
import fcntl
from scipy.signal import butter, filtfilt
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
from stockwell import st

def iqr_normalize(data):
    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    # IQR을 이용한 데이터 정규화
    normalized_data = (data - Q1) / IQR
    return normalized_data
#%%
def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

#%%
# detrend
def detrend(signal, Lambda):
    signal_length = len(signal)
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

#%%
# resample
def resample_signal(ori_signal, original_fs, target_fs):
    num_samples = int(len(ori_signal) * target_fs / original_fs)
    resampled_signal = resample(ori_signal, num_samples)
    return resampled_signal

#%% Zero phase filtered signal
def low_pass_filter(signal, cutoff, fs, numtaps=101):
    """
    FIR Low-pass filter를 적용하고 Zero-phase filtering을 통해 위상 지연을 제거하는 함수.
    Parameters:
    - signal: 입력 신호 (1D numpy array)
    - cutoff: 컷오프 주파수 (Hz)
    - fs: 샘플링 주파수 (Hz)
    - numtaps: 필터 계수의 수 (default=101)
    Returns:
    - filtered_signal: Zero-phase 필터링된 신호 (1D numpy array)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # FIR 필터 계수 생성
    fir_coeff = firwin(numtaps, normal_cutoff)

    # Zero-phase Filtering 적용
    filtered_signal = filtfilt(fir_coeff, [1.0], signal)

    return filtered_signal

#%%
# Stockwell 변환 함수
def stockwell_transform(signal, fmin, fmax, signal_length):
    df = 1. / signal_length
    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)
    trans_signal = st.st(signal, fmin_samples, fmax_samples)
    return trans_signal

#%% Main Code

# record_name = "sel30"  # 예: "100"
file_path = "./qt-database-1.0.0/"    # 예: "./mitdb/"
label_path="QTDB_segment_label"
p_label_path="QTDB_segment_P_label"
qrs_label_path="QTDB_segment_QRS_label"
t_label_path="QTDB_segment_T_label"
pre_processed_path="QTDB_preprocessed_signal"
s_transformed_path="QTDB_STransformed_npy"

with open(f'{file_path}/RECORDS', 'r') as file:
    file_names = file.readlines()  # 모든 줄을 리스트로 읽기
file_names=[file_name.strip() for file_name in file_names]
fmin = 0 # 0 hz for DC 
fmax = 15  #S transform max frequency
signal_length = 10 # 10 sec Signal

file_num=1

# ECG 데이터와 주석 읽어오기
for record_name in file_names:
    record = wfdb.rdrecord(f"{file_path}/{record_name}")  # .hea, .dat 파일 읽기
    print(f"{record_name} is processing... freq: {record.fs}...")
    annotation = wfdb.rdann(f"{file_path}/{record_name}", "pu0")  # .atr 파일 읽기
    signal = record.p_signal[:,0]
    signal = low_pass_filter(signal, 30, record.fs)
    signal = resample_signal(signal, record.fs, 100)
    signal = iqr_normalize(signal)
    signal = min_max_normalize(signal)
    labels = [0]*len(signal)
    qrscomplexs = [0 for i in range(len(signal))]
    pwaves = [0 for i in range(len(signal))]
    twaves = [0 for i in range(len(signal))]
    
    for symnum in range(len(annotation.symbol)):
        if annotation.symbol[symnum]=="(":
            start_point = int(annotation.sample[symnum]//2.5)
        elif annotation.symbol[symnum] in "Ntp":
            sym = annotation.symbol[symnum]
        elif annotation.symbol[symnum]==")":
            end_point = int(annotation.sample[symnum]//2.5)
            if sym == "N": # QRS complex
                for i in range(start_point, end_point+1):
                    qrscomplexs[i] = 1
                for i in range(start_point, end_point+1):
                    labels[i] = 2
            elif sym == "t": # Twave
                for i in range(start_point, end_point+1):
                    twaves[i] = 1
                for i in range(start_point, end_point+1):
                    labels[i] = 3
            elif sym == "p": # Pwave
                for i in range(start_point, end_point+1):
                    pwaves[i] = 1
                for i in range(start_point, end_point+1):
                    labels[i] = 1
            else:
                print("Something Error is occured >_< !!Help Me!!")
        else:
            print("Something Error is occured >_< !!Help Me!! 222")
            

    for i in range(0,len(signal)-1000,300):
        try:
            sig=signal[i:i+1000]
        except:
            break
        trans_signal = stockwell_transform(sig, fmin, fmax, signal_length)
        real_part = np.real(trans_signal)
        imag_part = np.imag(trans_signal)
        transformed_signal = np.stack((real_part, imag_part), axis=0)
        label=labels[i:i+1000]
        pwave=pwaves[i:i+1000]
        qrscomplex=qrscomplexs[i:i+1000]
        twave=twaves[i:i+1000]
        
        # plt.figure(figsize=(10,6))
        # plt.imshow(abs(trans_signal), origin="lower")
        # plt.show()
        
        # color_map = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
        # # 그래프 그리기
        # plt.figure(figsize=(10, 6))
        # # 구간별로 색상 지정하여 선 플롯
        # start = 0
        # for i in range(1, len(label)):
        #     if label[i] != label[start]:  # 레이블이 바뀌는 시점
        #         plt.plot(range(start, i), sig[start:i], color=color_map[label[start]], linewidth=2)
        #         start = i
        # plt.plot(range(start, len(label)), sig[start:], color=color_map[label[start]], linewidth=2)
        
        # # 그래프 레이블 및 범례 추가
        # plt.title('ECG Signal with Smooth Color Segments')
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.grid()
        
        # # 범례 추가
        # for label, color in color_map.items():
        #     plt.plot([], [], color=color, label=f'Label {label}')  # 빈 선으로 범례 생성
        # plt.legend()
        # plt.show()
        
        np.savetxt(f"./{pre_processed_path}/{file_num}.csv",sig,delimiter=",")
        np.save(f"./{s_transformed_path}/{file_num}.npy", transformed_signal)
        np.savetxt(f"./{label_path}/{file_num}_label.csv",label,delimiter=",")
        np.savetxt(f"./{p_label_path}/{file_num}_plabel.csv",pwave,delimiter=",")
        np.savetxt(f"./{qrs_label_path}/{file_num}_qrslabel.csv",qrscomplex,delimiter=",")
        np.savetxt(f"./{t_label_path}/{file_num}_tlabel.csv",twave,delimiter=",")
        
        file_num+=1
