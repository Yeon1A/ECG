# AIHolmes

---
## 1. 파라미터 설정
### 1) preprocess.json
해당 파일에서, 전처리 파라미터의 조정이 가능함.  
- **`task`**:  
  - `cls`: AAMI 기준에 따른 NSVFQ 분류 task  
  - `seg`: N, P, QRS, T Segmentation task
  
- **`dataset`** : 데이터셋 이름  
  - `mitdb` : MIT-BIH Arrhythmia Dataset (cls)
  - `QTDB` : QT Database v1.0.0 (seg)
  - `LUDB` : Lobachevsky University Electrocardiography Database (seg)

- **`input_dir`** : 원본 데이터셋이 저장된 위치 (zip 파일 해제 필요)  
  - if `dataset == "mitdb"` : ./Data/mit-bih-arrhythmia-database-1.0.0
  - if `dataset == "QTDB"` : ./Data/qt-database-1.0.0
  - if `dataset == "LUDB"` : ./Data/lobachevsky-university-electrocardiography-database-1.0.1/data

- **`output_dir`** : 전처리된 데이터 저장 경로  
  (이후, `train.json`의 `processed_data_dir`에서 재사용 예정)  
  - Default: `./Processed_Data`

- **`SMOTE`** : 데이터 불균형 보정을 위한 SMOTE 사용 여부  
  - `false` : 사용하지 않음  
  - `true`  : 전처리 이후 1D 신호 단계에서 SMOTE 사용  
    (⚠ 저장 용량 주의: 원본, SMOTE 및 변환본 모두 저장될 수 있음)

- **`window_seconds`** : 전처리 완료 시점에서 데이터 세그먼트의 길이 (초 단위)

- **`added_window_seconds`** : 한번 더 긴 윈도우 분할 길이 (초 단위)  
  (전처리 중 detrend 등 메모리 한계를 고려해 추가 윈도우를 사용)  
  → 기본적으로는 `window_seconds`보다 길거나 같아야 함.

- **`overlapped`** : window 겹침 비율(%)
  각 세그먼트가 몇 % 겹치도록 나누는지 결정  
  - 값은 **0 ~ 100 (%)**

  **예시 (윈도우 길이 10초 기준)**  
  - `0`  : `0 ~ 10`, `10 ~ 20` … (겹침 없음)  
  - `20` : `0 ~ 10`, `8 ~ 18` … (20% 겹침 → stride = 8초)  
  - `50` : `0 ~ 10`, `5 ~ 15`, `10 ~ 20` … (절반 겹침 → stride = 5초)
 
  **공식**
    
$$
\text{stride} = W \times \left(1 - \frac{\text{overlap}}{100}\right)
$$

- **`lpf`** : Low Pass Filter 적용 여부
  - **`lpf_hz`** : LPF cutoff frequency (30~40Hz 추천)
 
- **`downsample`** : Sampling Rate 조정 여부
  - **`downsample_hz`** : Sampling rate (100Hz 이상 추천)
 
- **`iqr`** : IQR(Interquartile Range) 기반 이상치 제거 적용 여부
  - **`iqr_ratio`** : 이상치 판단 기준 (IQR 범위를 몇 %로 둘 것인지)
 
- **`detrend`** : 신호의 baseline wander 제거 (선형/저주파 드리프트 제거)

- **`Stransform`** : Stockwell Transform 적용 여부(ECG 신호를 시간-주파수 2D representation으로 변환)
  - **`Stransform_fmin`** : Stockwell Transform 변환 시 최소 주파수 (Hz)
  - **`Stransform_fmax`** : Stockwell Transform 변환 시 최대 주파수 (Hz)
 
- **`Hilbert`** : Hilbert Transform 적용 여부

### 2) train.json

- **`task`** :
  - `cls` : AAMI 기준에 따른 NSVFQ **분류(Classification)** 학습 모드
  - `seg` : P/QRS/T 등 **세분화(Segmentation)** 학습 모드

- **`processed_data_dir`** :  
  - 전처리 결과 기본 폴더 경로 preprocess.json의 `output_dir`와 동일 (예: `./Processed_Data`)

- **`processed_data_idx`** :  
  - 사용할 하위 폴더 인덱스(전처리 시 Processed_Data0 등의 폴더 마지막 번호)  
  - 예: `Processed_Data3` 를 쓰려면 `3`
 
- **`model_name`** :  
  - 사용할 모델 이름 (예: `UNet1D`, `TCN`, `UNetQRS`, `ResNet2D` 까지 구현되어 있음.)

- **`weight_dir`** / **`weight_name`** :  
  - 가중치 저장/불러오기 경로와 파일명 (예: `./weights`, `model.pth`)
 
- **`epochs`** :  
  - 최대 학습 에폭 수
 

- **`learning_rate`** :  
  - 초기 학습률
 
- **`num_workers`** :  
  - Windows/디버깅 환경에서는 `0` 권장  
  - 값이 크면 로딩 속도는 빨라지지만, 환경에 따라 **프로세스(PID) 관련 오류**가 발생할 수 있음
 
- **`optimizer_name`** :  
  - 최적화 알고리즘 (`Adam`, `AdamW`, `SGD` 등)
 
- **`scheduler_enabled`** :  
  - 러닝레이트 스케줄러 사용 여부 (`true`/`false`)
 
- **`sclaer_enabled`** :  
  - **AMP(Mixed Precision)** 사용 여부  
  - `true`면 GradScaler로 half-precision 학습

- **`batch_size`** :  
  - 배치 크기
 
- **`patience`** :  
  - Early stopping / LR 스케줄러 등에서 개선 없을 때 대기 에폭 수
 
- **`inchannel`** :  
  - 입력 채널 수 (예: 1D 신호 1채널 → `1`, 실수/허수 2채널 → `2`)
 
- **`num_classes`** :  
  - 예측 클래스 수 (예: `4` → [N, S, V, Q], `5` → [N, S, V, Q, F], `4` → [N, P, QRS, T])
 
- **`file_dict`** :  
  - 클래스 이름 → 인덱스 매핑  
  - 예: `{ "N":0, "S":1, "V":2, "Q":3, "F":4 }`  
  - **seg task**에서는 무시 가능  
  - 데이터셋에 따라 특정 클래스(`F`)가 아예 없을 수 있음  
    → 이 경우 해당 키는 제외하고 사용

## 2. Custom
### 1) Dataset 추가 시
- **`Data/Data_config.py`**  
  다음 정보를 반환하도록 구성합니다.
  - `dataset` : `"QTDB"`, `"LUDB"`, `"mibdb"`
  - `task` : `cls`, `seg`
  - `record_names` : 학습에 사용할 레코드(파일) 목록
  - `original_fs` : 원본 샘플링레이트 (예: 360 Hz)
  - `annotation_word` : wfdb에서 해당 DB의 라벨 파일을 읽을 때 필요한 suffix
  - `target_channel` : multi-lead ECG 데이터 대비, 특정 리드 선택 (데이터셋마다 리드 순서가 다르므로 반드시 확인할 것)
  - `signal_length` : ECG 레코드 전체 길이 (초 단위)
 
- **`tools/Data_parser.py`**  
데이터셋의 annotation 형식을 읽어와 모델 학습에 맞는 표준 라벨 및 전처리 실행 모듈
  **주요 기능**
  - `label mapping`
  - `preprocessing`

해당 기능이 정상적으로 돌아가는 지 확인하고, 라벨 매핑이 기존 데이터셋과 다를 수 있으므로, 확인이 필요합니다

### 2) 원본 전처리 추가
- **`tools/preprocess_tools.py`** or **`tools/training_tools.py`**
  - 전체 전처리(저장용) 확장시, `tools/preprocess_tools.py` 에서 전처리 방식 추가 후, `tools/Data_parser.py` 에 적용
  - 일시 학습용 전처리 확장시, `tools/training_tools.py` 에서 Dataset 객체 생성 이후, `train.py` 에 적용 및 모델 채널 조정 등
 
### 3) 모델 추가
- **`tools/Models.py`**
  - 모델 추가는 `tools/Models.py`에서 진행
  - 추가 후, `tools/training_tools.py` 에서 `select_model` 확장

### 4) Optimizer 추가
- **`tools/training_tools.py`**
  - `tools/training_tools.py` 에서 직접 `select_optim` 확장

## 3. 실행
- **`main.py`** 에서 파일을 실행할 수 있음.
  - `python main.py preprocess` → 전처리 실행
  - `python main.py train` → 학습 실행

## 4. Result
### 1) Preprocess
- 각 Data_config 실행 결과는 `Processed_Data_info.xlsx` 에 자동 저장됩니다.
- 저장되는 정보는 각 실험 세팅(Idx)에 대한 전처리 및 데이터 구성 파라미터입니다.
- ⚠️ 주의사항
  - 데이터가 저장되는 폴더명 `Idx`는 `Processed_Data_info.xlsx`에서 현재까지의 최댓값 + 1 방식으로 부여됩니다.
  - 따라서 엑셀에서 가로줄(레코드)을 삭제하거나 Idx를 직접 수정할 경우, 이후 생성되는 데이터가 기존 폴더를 덮어쓸 위험이 있습니다.
  - 안전한 관리를 위해 엑셀 파일을 직접 수정하지 않는 것을 권장합니다.
 
### 2) Train
- 학습 결과는 Task 유형(cls, seg) 에 따라 구분되어 저장됩니다.
  - 분류(Classification) → `cls_result_dict.xlsx`  
  - 분할(Segmentation) → `seg_result_dict.xlsx` 
- 결과 파일에는 사용된 전처리 방식(Processed_Data_info.xlsx의 Idx 기반) 과 평가 지표가 함께 기록됩니다.
- Classification(cls) 과 Segmentation(seg) 모두 공통적으로 주요 지표(Accuracy, Loss)가 저장되며,
- Classification는 F1, PPV, Se, Kappa, MCC가 포함되고, Segmentation은 IOU 등의 지표가 포함될 수 있습니다.
