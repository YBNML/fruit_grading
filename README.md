# fruit_grading — Fruit Quality ML Pipeline

과수 선별기(sorter)의 3대 카메라 영상으로 **과일 품질을 자동 분류·회귀**하는 end-to-end
ML 파이프라인. 복숭아(천중도)와 감귤(황금향) 두 작물을 개별 모델로 학습한다.

원본 이미지 데이터셋과 학습된 모델 가중치는 공개 범위에 포함하지 않았다. 본 저장소는
**코드 + 학습 설정 + 성능 결과(history)** 만 공개하며, 재현은 동일 형식의 데이터셋을
준비하면 가능하도록 구성했다.

---

## 1. 문제 정의

농가에서 수집된 과일의 **품질 등급을 3단계로 자동 분류**한다. 기존 수작업 선별을
대체하기 위해, 선별기에 부착된 3대 oCam(top1 / top2 / bottom1)의 촬영 이미지를
입력으로 받는다.

라벨이 되는 "품질" 자체의 공식 정의가 원 데이터에 없어서 **측정값(weight, height,
max_w, min_w) 의 per-crop 3분위 (tertile)** 로 3등급을 합성했다. 이것이 Phase 0.
이어서 측정값 자체를 회귀로 맞추는 Phase 1으로 확장했으며 (등급보다 정보량이 많음),
이후 Phase 2(멀티태스크), Phase 3(3뷰 통합), Phase 4(황금향 세그멘테이션),
Phase 5(brix 회귀)로 확장 예정.

## 2. 데이터 (요약)

| 작물 | 촬영일 수 | 과일 수 (3뷰 × N) | 이미지 수 | 라벨 포맷 |
|---|---|---|---|---|
| 복숭아 천중도 | 9일 (2023-08) | 1,278 | 3,834 | 날짜별 JSON (measurement / marketability / camera / farm) |
| 감귤 황금향 | 7일 (2023-11~12) | 2,309 | 6,927 | 날짜별 CSV (CP949/UTF-8 혼재) + labelme 폴리곤 JSON |

- 동일 과일당 3뷰(top1 / top2 / bottom1) 이미지 → **fruit_id = (date, index)** 로 그룹
- 측정값: brix(당도), weight(g), height(mm), max_w/min_w(mm)
- 황금향은 `labelme` 포맷의 폴리곤 라벨도 동봉 — 세그멘테이션 태스크에 활용 예정

raw 데이터셋은 본 저장소에 포함하지 않는다. `make_labels.py` 가 DB/ 의 원본 라벨을
파싱해 훈련용 `label_{crop}.csv` 를 생성하는 스크립트.

## 3. 파이프라인 구조

```
DB/peach_천중도/{date}/{t1,t2,b1}/*.jpeg   ─┐
DB/귤_황금향/{date}/{t1,t2,b1}/*.jpeg      ─┤  make_labels.py
DB/peach_천중도/{date}/*_labeling_01.json ─┤  ───────────────────>  willmer/label_{crop}.csv
DB/귤_황금향/{date}/labeling.csv           ─┘

willmer/label_{crop}.csv  →  train.py  →  runs/{exp}/best.pt  +  history.json
```

### 3.1 라벨 빌드 (`make_labels.py`)

- 작물마다 다른 라벨 포맷(JSON vs CSV, UTF-8 vs CP949)을 통합 스키마로 정규화
- 경로는 촬영 머신마다 프리픽스가 달라서 `"database"` 세그먼트를 앵커로 잘라내 재매핑
- `weight` 3분위를 기본 grade 기준으로 사용 — `--grade-field` 로 다른 측정값 선택 가능

### 3.2 학습 (`train.py`)

주요 설계 결정:

- **ResNet50 전이학습** (ImageNet V2 가중치) + 마지막 head 교체
  (분류: 3 logits / 회귀: 1 scalar)
- **Fruit-level split**: 같은 과일의 3뷰가 train/test에 동시에 들어가면 성능이
  부풀려지므로, `(date, fruit_idx)` 단위로 그룹해 70/30 분할
- **Fruit-level evaluation**: 3뷰 예측을 평균해 **배포 환경과 동일한 조건으로 재평가**
  (선별기는 항상 3뷰를 동시에 관측). 샘플 단위 대비 대체로 MAE 30~40% 감소
- **Early stopping** (patience=5 기본) + **Cosine LR annealing**
- 분류: CrossEntropy + `WeightedRandomSampler` (클래스 불균형 대비, 현재는 tertile이라
  거의 균형이지만 다른 grade field로 전환 시 유용)
- 회귀: `SmoothL1Loss` (이상치에 CF MSE보다 덜 민감), MAE/R²/MAPE 리포트
- **MPS 지원**: Apple Silicon(M4)에서 GPU 가속. float64 미지원 이슈 대응 (float32 캐스팅)

### 3.3 입력 전처리

```
ToPILImage → CenterCrop(560) → Resize(224, 224) →
  [train only] RandomHorizontalFlip(0.5) + RandomVerticalFlip(0.2)
             + ColorJitter(brightness/contrast/saturation = 0.1)
→ ToTensor → Normalize(ImageNet mean/std)
```

CenterCrop 560 — oCam 720px 세로 영상에서 과일을 중앙에 두고 배경을 잘라냄.
ImageNet normalization은 ResNet50 가중치 재사용을 위해 유지.

## 4. 성능 결과

전 실험 공통: ResNet50 pretrained, SGD(0.9, 1e-4), Cosine LR, 20 epoch, patience 5.

### 4.1 3-grade Classification (Phase 0 baseline)

품질 등급은 weight 3분위 (per-crop) 로 합성.

| 작물 | best val_acc | best epoch | N (train / test) |
|---|---|---|---|
| 복숭아 | **81.72 %** | 28 (30ep) | 2,685 / 1,149 |
| 황금향 | **92.82 %** | 25 (30ep) | 4,845 / 2,082 |

복숭아 confusion (best epoch, rows=true, cols=pred):
```
       pred=0  pred=1  pred=2
true=0   362     24      1    (93.5%)  ← 소
true=1    83    252     67    (62.7%)  ← 중  (경계 모호)
true=2     1     34    325    (90.3%)  ← 대
```

중간 등급이 양옆으로 새어나가는 현상 — tertile 경계 근처의 라벨 본질적 모호성.
회귀(Phase 1)로 전환하면 이 문제가 사라짐.

### 4.2 Regression (Phase 1)

| 작물 | target | sample MAE | sample R² | sample MAPE | **fruit MAE** | **fruit R²** | **fruit MAPE** |
|---|---|---|---|---|---|---|---|
| 복숭아 | weight | 12.22 g  | 0.955 | 3.05% | **7.61 g** | **0.983** | **1.89%** |
| 복숭아 | height | 2.24 mm  | 0.798 | 2.73% | **1.92 mm** | **0.843** | **2.35%** |
| 복숭아 | max_w  | 1.37 mm  | 0.944 | 1.44% | **1.07 mm** | **0.967** | **1.14%** |
| 복숭아 | min_w  | 1.50 mm  | 0.908 | 1.69% | **1.24 mm** | **0.938** | **1.41%** |
| 황금향 | weight | 4.58 g   | 0.909 | 3.17% | **3.55 g**  | **0.923** | **2.52%** |
| 황금향 | height | 1.49 mm  | 0.732 | 2.57% | **1.26 mm** | **0.773** | **2.18%** |
| 황금향 | max_w  | 1.04 mm  | 0.896 | 1.51% | **0.85 mm** | **0.918** | **1.23%** |
| 황금향 | min_w  | 0.86 mm  | 0.943 | 1.28% | **0.69 mm** | **0.963** | **1.03%** |

주요 관찰:

- **R² 만으로 평가하면 안 됨.** height는 MAPE 기준으론 weight보다 정확(2.7% vs 3.1%)
  하지만 R²는 낮음(0.80 vs 0.96). 타겟 분산 자체가 작을 때(65~100mm) R²가 패널티를
  받는 구조. 운용 기준은 **MAPE + MAE** 가 더 적절.
- **Fruit-level 집계로 거의 공짜 점수**: 복숭아 weight 샘플 MAE 12.2g → fruit MAE 7.6g
  (38% 감소). 3뷰가 서로 다른 각도의 노이즈를 상쇄.
- 실용적으로는 **MAPE 2~3%, R² 0.95 수준**이 선별기 운용에 충분하다고 판단.
- **데이터 정제 중요성 확인**: 황금향 weight 초기 학습에서 R² 0.18이 나왔는데, 원인은
  20231202 날짜의 18개 행(전체 0.26%)에 weight=1210g이라는 비현실적 값이 들어있어서.
  박스 무게가 입력된 것으로 추정. `SANITY_BOUNDS`로 상식 범위 벗어난 값을 None 처리하니
  R²가 0.18 → 0.92로 회복. 소수 outlier의 SS_total 지배 효과 교과서적 사례.

### 4.3 Multi-View Regression (Phase 3)

Phase 1 은 단일 이미지로 학습 후 **추론 시에만** 3뷰 예측을 평균(후처리). Phase 3 은
모델이 처음부터 3뷰를 **동시에** 입력받는 구조 — ResNet50 backbone을 공유해서 t1/t2/b1
각각의 feature vector(2048-dim)를 뽑고, 순서대로 concat한 뒤 Linear head로 예측.

| 작물 | target | **MV MAE** | MV R² | MV MAPE | vs Phase 1 | 승자 |
|---|---|---|---|---|---|---|
| 복숭아 | weight | **6.78 g**  | 0.988 | 1.78% | -10.9% | **MV** |
| 복숭아 | height | 1.98 mm     | 0.842 | 2.42% | +2.9%  | SV |
| 복숭아 | max_w  | **0.96 mm** | 0.973 | 1.02% | -10.0% | **MV** |
| 복숭아 | min_w  | **1.22 mm** | 0.936 | 1.37% | -1.5%  | MV |
| 황금향 | weight | 3.89 g      | 0.922 | 2.68% | +9.6%  | SV |
| 황금향 | height | **1.23 mm** | 0.778 | 2.14% | -2.3%  | MV |
| 황금향 | max_w  | **0.69 mm** | 0.933 | 1.00% | **-19.3%** | **MV** |
| 황금향 | **min_w** | **0.57 mm** | **0.975** | **0.85%** | **-17.9%** | **MV** 🏆 |

**6/8에서 MV 우세**, 특히 **황금향 너비(max_w/min_w) 에서 18% 대폭 개선**. 현재까지 최고
결과는 **황금향 min_w MV: MAPE 0.85%, R² 0.975**.

주요 관찰:

- **너비(max_w/min_w)에서 MV 효과가 큼** — 3뷰가 서로 다른 각도의 최대/최소 너비를
  관측하므로 모델이 암묵적 3D 재구성 학습 가능. 단일뷰 평균은 이 정보를 잃어버림
- **높이(height)에서는 MV 이점 미미** — 높이는 어느 뷰에서도 비슷하게 보여 상보 정보 적음
- **황금향 weight만 SV가 앞섬(+9.6%)** — MV가 epoch 11에 early stop. full 20 epoch 돌렸다면
  결과가 다를 수 있음. 추가 실험으로 검증 가치 있음
- **파라미터/연산 비용**: backbone 공유라 파라미터는 head 확장분(3×2048 vs 2048)만 늘어남.
  학습 시간도 single-view와 거의 동일 (같은 총 이미지를 처리)

## 5. 재현 방법

### 5.1 환경 설치

```bash
conda create -n fruit_grading python=3.11 -y
conda activate fruit_grading
pip install -r willmer/requirements.txt
```

확인: Apple Silicon M시리즈에서 MPS 백엔드 정상 동작. CUDA GPU도 자동 감지.

### 5.2 데이터 준비

raw 이미지/라벨을 `DB/{peach_천중도,귤_황금향}/{date}/{t1,t2,b1,image}/` 구조로 배치.
본 저장소엔 데이터셋이 포함되지 않았으므로, 자신의 수집 데이터를 동일 구조로
맞추거나 `make_labels.py` 의 파서 함수(`load_peach`, `load_tangerine`)를
커스터마이즈해야 한다.

### 5.3 라벨 CSV 생성

```bash
cd willmer
python make_labels.py                          # weight 기반 grade
python make_labels.py --grade-field height     # height 기반으로 변경 시
```

산출: `willmer/label_peach.csv`, `willmer/label_tangerine.csv` (저장소에 커밋됨).

### 5.4 학습

```bash
# 3-grade classification
python train.py --label label_peach.csv --out runs/peach_cls --task cls --target grade

# regression (continuous measurement)
python train.py --label label_peach.csv     --out runs/peach_weight --task reg --target weight
python train.py --label label_tangerine.csv --out runs/tan_height   --task reg --target height \
                --model densenet201
```

주요 옵션: `--epochs 20 --patience 5 --batch-size 16 --lr 1e-3 --mask-bg`.
생성물: `runs/{exp}/best.pt`(무시됨) + `runs/{exp}/history.json`(공개).

Phase 3 (multi-view) 는 별도 스크립트:

```bash
python train_mv.py --label label_peach.csv --out runs_mv/peach_weight --task reg --target weight
```

차이점: 샘플 단위가 과일 1개(3뷰 stack); `--batch-size 8` 기본값 (샘플당 이미지 3배).
`MultiViewModel` 은 backbone을 3번 공유 forward하여 feature concat 후 head로 예측.

## 6. 기술 스택

- **Framework**: PyTorch 2.11, torchvision 0.26
- **Backbone**: ResNet50 (ImageNet V2 pretrained) / DenseNet201 (옵션)
- **Data ops**: NumPy, OpenCV, Pillow
- **Optim**: SGD + momentum, Cosine annealing, SmoothL1 / CrossEntropy
- **Runtime**: Apple MPS / CUDA 자동 선택
- **Python**: 3.11

## 7. 로드맵

- [x] **Phase 0** 3-grade classification baseline
- [x] **Phase 1** 4-target regression (weight / height / max_w / min_w)
- [x] **Phase 3** Multi-view 통합 모델 (3뷰 feature concat) — 6/8 target에서 SV 우세
- [ ] **Phase 2** Multi-task (한 backbone 공유 + 4 heads)
- [ ] **Phase 4** 황금향 세그멘테이션 (labelme → YOLOv8-seg / Mask R-CNN)
- [ ] **Phase 5** Brix 회귀 (내부 성질, 난이도 ↑)

---

## Notes

- 배경 마스킹(`--mask-bg`): 기존 선별기의 파란 배경에 맞춰 튜닝된 HSV threshold.
  새 촬영 환경에선 thresholds 재조정 필요. Phase 1에선 사용 안 함.
- 라벨 CSV 인코딩: 황금향 `labeling.csv` 는 날짜에 따라 CP949/UTF-8 혼재 —
  `make_labels.py` 에서 양쪽 모두 시도하는 fallback 처리.
- 본 저장소는 실험 기록용이며, 실제 선별기 배포 코드는 별도 리포(ROS1 기반, 비공개).
