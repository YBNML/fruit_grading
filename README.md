# fruit_grading — Fruit Quality ML Pipeline

과수 선별기(sorter)의 3대 카메라 영상으로 **과일 품질을 자동 분류·회귀**하는 end-to-end
ML 파이프라인. 복숭아(천중도)와 감귤(황금향) 두 작물을 개별 모델로 학습한다.

원본 이미지 데이터셋과 학습된 모델 가중치는 공개 범위에 포함하지 않았다. 본 저장소는
**코드 + 학습 설정 + 성능 결과(history)** 만 공개하며, 재현은 동일 형식의 데이터셋을
준비하면 가능하도록 구성했다.

---

## 📌 Executive Summary (TL;DR)

| 항목 | 수치 |
|---|---|
| 총 실험 수 | **26 runs** (5 Phase + backbone 비교) |
| 누적 학습 시간 | ~20시간 (Apple M4 MPS) |
| 모델 개수 (운용) | 작물당 4개 (target별) × 2작물 = 8 모델 |
| **최고 성능** | 황금향 min_w MV — **MAPE 0.85%, R² 0.975, MAE 0.57mm** |
| 크기 예측 (8 target) | **MAPE 1~3%, R² 0.84~0.99** |
| Brix(당도) 예측 | **MAPE 2.6~3.6%, R² 0.60~0.68** |

### 핵심 결론
1. **멀티뷰 구조가 가장 우수** — 단일뷰 평균 대비 6/8 target에서 승 (최대 19% 개선)
2. **멀티태스크는 역효과** — 8/8 target 후퇴 (통설 반증)
3. **내부 성질(당도)도 외형으로 예측 가능** — MAPE 2.6% 달성
4. **소규모 데이터에선 CNN > Transformer** — ViT-B/16 vs ResNet50: ResNet50 4/4 승

---

## 1. 문제 정의

기존 선별 프로세스는 과일마다 **전자저울(무게) + 캘리퍼스(치수) + 비파괴 당도계(brix)**
를 순차로 거쳐 **전수 측정**하는 구조다. 측정 정확도는 높지만 과일 1개당 3개 이상의
장비를 경유해야 해서 **처리 속도·작업자 피로·장비 비용**이 병목이다.

본 프로젝트의 목표는 **선별기에 이미 부착된 3대 oCam(top1 / top2 / bottom1)의
영상만으로 동일 측정값을 회귀 예측**하는 것이다. 별도 저울/캘리퍼스/당도계 없이
선별기 흐름 중에 자동 측정·등급화가 완료되는 단일 파이프라인을 목표로 한다.

라벨이 되는 "품질 등급"은 원 데이터에 공식 정의가 없어서 **측정값(weight, height,
max_w, min_w) 의 per-crop 3분위 (tertile)** 로 3등급을 합성했다 (Phase 0). 이어서
측정값 자체를 회귀로 맞추는 Phase 1 으로 확장했으며 (등급보다 정보량이 많음),
이후 Phase 2(멀티태스크), Phase 3(멀티뷰), Phase 5(brix), 그리고 backbone 비교
(ResNet50 vs ViT) 까지 수행했다. Phase 4(세그멘테이션) 는 skip.

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
| 복숭아 | brix   | 0.599 °Bx | 0.513 | 4.51% | **0.512 °Bx** | **0.639** | **3.86%** |
| 황금향 | brix   | 0.348 °Bx | 0.485 | 2.98% | **0.308 °Bx** | **0.593** | **2.64%** |

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
- **Brix(당도) 는 내부 성질인데도 의외로 예측 가능**: 크기 target 대비 R² 는 낮지만
  (0.60~0.64) MAPE 는 2.6~3.9% 수준으로 실용 범위. 과일 껍질 색/질감이 당도와
  중간 정도 상관이 있음을 시사. Peach R² 0.64가 황금향 R² 0.59보다 높은 이유는
  peach brix 분포가 더 넓어서(10.4~16.6, σ 1.15 vs 황금향 10.2~13.5, σ 0.62)
  신호 대 잡음비가 좋기 때문.

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
| 복숭아 | brix   | **0.47 °Bx** | 0.676 | 3.57% | -7.8% | MV |
| 황금향 | brix   | **0.30 °Bx** | 0.629 | 2.57% | -3.2% | MV |

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

### 4.4 Multi-View + Multi-Task Regression (Phase 2)

MV backbone을 그대로 두고 마지막 `Linear` head만 (3·2048) → 4 로 확장. 4 개
target(weight/height/max_w/min_w) 을 하나의 모델이 동시에 출력. Target 스케일이
달라서 (weight ~400g vs height ~80mm) z-score 정규화 공간에서 학습하고 추론 시
denormalize. Target 통계(mean/std)는 checkpoint에 buffer로 저장.

| 작물/target | P3 MV | **P2 MV+MT** | Δ (vs P3) |
|---|---|---|---|
| peach/weight | 6.78 g  | 12.29 g | +81.3% |
| peach/height | 1.98 mm | 2.28 mm | +15.4% |
| peach/max_w  | 0.96 mm | 1.47 mm | +52.8% |
| peach/min_w  | 1.22 mm | 1.41 mm | +15.4% |
| tangerine/weight | 3.89 g  | 5.15 g  | +32.4% |
| tangerine/height | 1.23 mm | 1.55 mm | +25.6% |
| tangerine/max_w  | 0.69 mm | 0.89 mm | +30.1% |
| tangerine/min_w  | 0.57 mm | 0.77 mm | +36.6% |

**결과: 8/8 target에서 Phase 2가 Phase 3보다 후퇴** — "멀티태스크가 feature 공유로
약한 target을 돕는다"는 가설이 이 데이터/구조에선 성립하지 않음.

추정 원인:

- **Head 병목**: 하나의 Linear head가 4 target 동시 서비스 — 각 target에 특화된 기울기가
  서로 간섭. Phase 3 는 target마다 dedicated Linear head를 씀
- **Target correlation**: 4 target 전부 "크기" 계열이라 feature 공유 이점은 작고
  capacity 부족만 크게 체감
- **Early stop 미발동**: 두 실험 모두 20 epoch full run. 수렴은 도달했지만 최적해 자체가
  single-task보다 낮음 → 더 긴 학습으로 좁혀지기 어려움

**개선 여지** (시도하지 않음): head를 MLP(2-layer)로 확장 / task-specific head 추가 /
uncertainty-weighted loss / task별 gradient balancing. 그러나 Phase 3 성능이 이미 실용
요구를 넘어서 (MAPE 1~3%) 추가 엔지니어링의 투자 대비 이득은 제한적.

**운용 권장**: **Phase 3 (MV + 단일태스크 4개 모델)** 이 가장 정확. 모델 관리 비용이
고민이면 Phase 2 를 2 모델(크롭당 1) 로 쓸 수도 있으나 MAPE 1.5%p 정도 손해 감안.

### 4.5 Backbone 비교: ViT-B/16 vs ResNet50 (Phase 3 구조 위에서)

**CNN(ResNet50) vs Transformer(ViT-B/16)** 비교. Phase 3(MV) 구조 고정, backbone만
교체. 두 대표 target(weight / brix) × 2 작물 = 4 실험. 같은 optimizer(SGD mom=0.9,
lr=1e-3), 같은 epoch(20 + patience 5), ViT batch=4 (메모리).

| target | ViT MAE | ResNet50 MAE | ΔMAE | ViT R² | R50 R² |
|---|---|---|---|---|---|
| peach weight     | 60.71 g   | **6.78 g**   | +795.5% | -0.008 | **0.988** |
| peach brix       | 0.613 °Bx | **0.472 °Bx** | +29.9%  | 0.473  | **0.676** |
| tangerine weight | 4.213 g   | **3.890 g**  | +8.3%   | 0.914  | **0.922** |
| tangerine brix   | 0.392 °Bx | **0.298 °Bx** | +31.4%  | 0.355  | **0.629** |

**ResNet50 이 4/4 전승**. 격차는 +8% (tangerine weight, 근접) 에서 +796% (peach weight,
완전 실패) 까지 target-조합별로 큼.

주요 관찰:

- **peach weight ViT 의 완전 실패**: R² -0.008 은 모델이 본질적으로 평균만 예측한다는
  뜻. Early stop 이 epoch 12 에서 발동 — 학습 곡선이 plateau 에 갇힘. SGD+lr=1e-3 이
  CNN 기준 세팅이라 ViT 에게 불리했을 가능성 큼. 같은 recipe 로 돌린 다른 3 실험은
  정상 학습됐으니 특정 시드/초기화 조합 이슈일 수도 있음
- **데이터가 많은 쪽이 ViT 에 유리**: 황금향(n=2,309 과일) > 복숭아(n=1,278). 황금향
  weight 는 ViT 가 CNN 과 8% 격차까지 좁힘. ViT 가 data-hungry 하다는 통설 검증
- **Brix 태스크에서도 CNN 우세**: "ViT 의 global attention 이 외형 기반 내부성질 예측에
  유리할 수 있다" 는 가설은 지지되지 않음. 두 작물 모두 ViT 가 ~30% 더 나쁨
- **"공정 비교 recipe" 의 한계**: ViT 는 관례상 AdamW + lr 1e-4~1e-5 + warmup +
  cosine LR + batch size 64+ 가 표준. 같은 SGD recipe 로 비교한 건 CNN 편향. 별도
  튜닝으로 격차 줄일 여지 있음 — 그러나 "CNN 기준 recipe 에서 CNN 이 압승" 자체가
  엔지니어링 결정 근거로 의미 있음

**운용 권장**: 이 데이터셋/예산 조건에선 ResNet50 이 명확한 기본 선택. ViT 도입은
(1) 데이터가 수만 단위로 늘거나, (2) ViT 전용 recipe 로 재학습 가능할 때 재검토.

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

Phase 2 (multi-view + multi-task) 는 또 다른 스크립트:

```bash
python train_mv_mt.py --label label_peach.csv --out runs_mv_mt/peach
```

4 target(weight/height/max_w/min_w) 을 동시 예측. z-score 정규화는 trainer 에서
train split 기준으로 계산, 모델 buffer(`target_means`, `target_stds`)에 저장.

## 6. 기술 스택

- **Framework**: PyTorch 2.11, torchvision 0.26
- **Backbone**: ResNet50 (ImageNet V2 pretrained) / DenseNet201 (옵션)
- **Data ops**: NumPy, OpenCV, Pillow
- **Optim**: SGD + momentum, Cosine annealing, SmoothL1 / CrossEntropy
- **Runtime**: Apple MPS / CUDA 자동 선택
- **Python**: 3.11

## 7. Key Findings & Insights

### 7.1 실증적 발견

**(1) Fruit-level 집계만으로 평균 38% 오차 감소**
동일 과일의 3뷰 예측을 평균하는 것만으로도 큰 이득이 있다. 추가 학습 없이 free win.
예: 복숭아 weight 샘플 MAE 12.2g → fruit MAE 7.6g.

**(2) 멀티뷰 학습(Phase 3)이 단일뷰+평균(Phase 1)을 6/8 target에서 우위**
너비(max_w/min_w)에서 가장 큰 이득(18~19% 개선). 3뷰가 서로 다른 각도에서
최대/최소 너비를 관측하므로 모델이 암묵적으로 3D 형태를 재구성.
높이는 어느 뷰에서도 비슷하게 보여 상보 정보 적고 이득 미미.

**(3) 멀티태스크는 본 데이터셋에서 역효과** (8/8 후퇴)
단일 Linear head가 4 target을 동시에 서비스할 때 gradient 간섭이 feature 공유
이득을 초과. 모든 target이 "크기" 계열이라 공유가 주는 regularization도 약함.
**통설("멀티태스크는 약한 target을 돕는다")의 반례 사례**로 포트폴리오 가치 있음.

**(4) 외형 이미지로 Brix(당도) 예측 가능**
R² 0.60~0.68, MAPE 2.6~3.9%. 과일 껍질 색/질감이 당도와 상관이 있어 CNN이
학습 가능. 근적외선 센서 없이도 실용 범위 달성.

**(5) 소규모 데이터셋(N≈1.3k~2.3k 과일)에선 CNN > Transformer**
같은 recipe(SGD+lr=1e-3)로 비교 시 ResNet50이 ViT-B/16을 4/4 승. ViT는 더 많은
데이터 + 전용 optimization recipe 필요.

**(6) 데이터 품질이 성능 지배 요인**
황금향 weight 학습에서 R² 0.18이 나왔는데 원인은 전체 0.26%에 불과한 outlier
18건(weight=1210g, 박스 무게 오입력 추정). 상식 범위 필터링 후 R² 0.92로 복구.
**적은 수의 outlier가 SS_total을 지배하는 교과서적 사례**.

**(7) R² 단독 지표는 misleading**
target 분산이 작을수록 R²가 패널티를 받음. 예: 복숭아 height는 MAPE 2.73%인데
R² 0.80. 운용 지표는 **MAPE + MAE 병행 검토**가 적절.

### 7.2 엔지니어링 결정 근거

| 결정 | 근거 |
|---|---|
| ResNet50 기본 선택 | ViT 대비 4/4 승, 학습 안정성, 작은 파라미터로 과적합 위험↓ |
| MV 단일태스크 (4 모델/crop) | 6/8 최고 성능, 단순 구조, 배포/디버깅 용이 |
| Fruit-id 단위 split | 같은 과일 3뷰 누수 방지 — 이것 없이는 val 지표 부풀려짐 |
| SmoothL1 Loss | weight 범위 넓음(황금향 72~237g) + 소수 outlier 대응 |
| Cosine LR + early stop | 대부분 15~20 epoch 내 수렴, patience 5로 자동 중단 |
| `SANITY_BOUNDS` | 상식 범위 기반 데이터 정제가 복잡한 모델 튜닝보다 ROI 높음 |

---

## 8. Limitations & Future Research Directions

### 8.1 현재 한계

- **데이터 규모**: 작물당 1,278 / 2,309 과일. 계절별 / 품종별 / 농가별 변동성 반영 불가
- **2 작물만**: 복숭아 천중도 + 감귤 황금향. 타 품종(유명, 박애, 황도 등) 일반화 검증 없음
- **단일 촬영 환경**: 동일 선별기 + 동일 조명. 도메인 차이 강건성 미검증
- **라벨 노이즈 일부 잔존**: SANITY_BOUNDS로 극단치만 제거. 경계 부근 측정 오차는 남음
- **세그멘테이션 미활용**: 황금향 labelme 폴리곤이 있으나 Phase 4 미실행
- **Brix R²가 한계**: 0.60~0.68 수준. 외형 신호만으론 상한이 있을 가능성 (분광/NIR 병용 필요)
- **해상도 224x224**: ImageNet 표준이라 미세한 결함/표면 특징 손실 가능
- **ViT 불공정 비교**: 동일 recipe 기준이라 ViT에 불리. AdamW+warmup 미시도

### 8.2 단기 개선 (구현 가능)

1. **Phase 4 세그멘테이션** — 황금향 labelme 폴리곤으로 YOLOv8-seg 학습 → tight crop
   → 기존 분류/회귀기 재학습 (기대 이득: MAPE 5~15% 추가 감소)
2. **ViT 전용 recipe로 재비교** — AdamW + lr 1e-4 + warmup + batch 16+ + LayerNorm 학습
3. **Test-Time Augmentation (TTA)** — 추론 시 flip/rotate 여러 번 평균 (기대 2~5% 추가)
4. **앙상블** — seed 3~5개 평균, 또는 ResNet50+DenseNet201 결합
5. **해상도 스케일업** — 224 → 320 또는 384, 특히 세부 표면 feature 필요한 brix에 도움
6. **Multi-task 아키텍처 개선** — 공유 trunk 후 per-target head (2-layer MLP), uncertainty
   weighting / GradNorm으로 gradient 균형화
7. **Attention 시각화** — GradCAM / ScoreCAM으로 "모델이 어디를 보고 있는지" 해석 가능성
8. **모델 경량화** — MobileNet / EfficientNet 전환, ONNX/CoreML 변환으로 엣지 배포
9. **Multi-view Attention Fusion** — 현재 concat 대신 **learnable weighted average**
   (softmax(α_t1, α_t2, α_b1) · f_i)로 뷰별 기여도를 명시적으로 학습. 파라미터 3개
   추가만으로 "어느 뷰가 어느 target에 가장 중요한가" 해석 가능 + 뷰 품질 편차
   있을 때 적응적 가중치 부여. 기대 이득: MAE 1~3% 추가 감소 + 해석성 확보.
   Self-attention(Transformer-style)은 파라미터 >10M 추가라 현 데이터 규모엔 risky,
   weighted average부터 시도하는 게 ROI 높음.

### 8.3 중기 연구 (추가 데이터/인프라 필요)

1. **데이터 확장**
   - 여러 계절/해 (내년도 수확 포함)
   - 다품종 (복숭아 10+ 품종, 감귤 5+ 품종)
   - 다농가 (지리/재배조건 분산)
   - 상처/병반 등 부정적 케이스 데이터셋 추가
2. **도메인 적응(Domain Adaptation)**
   - 카메라/조명 변경 시 재학습 없이 배포 가능성
   - Adversarial training, feature alignment, test-time adaptation 실험
3. **자가지도학습(Self-supervised)**
   - 비라벨 과일 이미지 대량 활용. DINOv2 / MAE pretrain → 소량 라벨로 fine-tune
   - 라벨링 비용 절감 + 더 강건한 feature
4. **반지도학습(Semi-supervised)**
   - 측정값 없이 촬영만 된 이미지 활용 (FixMatch, pseudo-labeling)
5. **Active Learning**
   - 모델이 가장 불확실한 샘플을 우선 측정 의뢰
   - 동일 성능 기준 50% 이하 라벨링 비용으로 도달 가능성

### 8.4 장기 연구 (센서/시스템 수준)

1. **멀티모달 융합** — RGB + NIR(근적외선) + 하이퍼스펙트럴
   - 특히 brix 정확도 R² 0.9+ 로 끌어올릴 잠재력 (비접촉 brix 측정 논문 다수)
2. **비디오/시퀀스 입력** — 롤러에서 굴러갈 때 연속 프레임 활용
   - 정지 3뷰보다 많은 정보. Video Transformer / ConvLSTM
3. **3D 형태 재구성** — SfM / NeRF 기반 과일의 완전한 3D 모델 → 부피/표면 계산 직접화
4. **결함 세그멘테이션 + 등급화 통합** — 단일 파이프라인으로 크기/당도/상품성 동시 산출
5. **품종 판별 + 측정** — 현재는 품종을 아는 전제. open-set 품종 인식 통합
6. **Cross-crop 전이** — 복숭아 모델을 살구/자두에 few-shot 전이

### 8.5 운용(Operations) 과제

- 실시간 처리 latency (현재 선별기 타겟: <100ms per fruit)
- 온디바이스 vs 엣지 서버 trade-off
- 모델 갱신 주기 (계절별 drift)
- 라벨 수집 표준화 (측정 오차 <1% 목표)
- 운용자 대시보드 (모델 신뢰도 경고, 이상치 탐지)

---

## 9. 로드맵 (완료/남음)

- [x] **Phase 0** 3-grade classification baseline
- [x] **Phase 1** 4-target regression (weight / height / max_w / min_w) — SV + fruit-avg
- [x] **Phase 2** Multi-view + Multi-task — 전 target Phase 3 대비 후퇴 (통설 반증 케이스)
- [x] **Phase 3** Multi-view 통합 모델 (3뷰 feature concat) — 6/8에서 SV 우세, **현재 best**
- [x] **Phase 5** Brix(당도) 회귀 — MV MAPE 2.6~3.6%
- [x] **Backbone 비교** ResNet50 vs ViT-B/16 — CNN 4/4 승 (같은 recipe 기준)
- [ ] **Phase 4** 세그멘테이션 — skip 결정
- [ ] ViT 전용 recipe 재검증 (AdamW + warmup)
- [ ] TTA / 앙상블 / 해상도 스케일업 실험
- [ ] Attention 해석 / 모델 경량화

---

## 프로젝트 요약 PPT

전체 프로젝트를 19 슬라이드로 정리한 포트폴리오 덱:
- 파일: `fruit_grading_summary.pptx` (저장소 루트)
- 재생성: `pip install python-pptx matplotlib && python make_ppt.py`
- 내용: 문제 정의 → 데이터 → 파이프라인 → 5 Phase 결과 → 인사이트 → 한계 →
  단기·중기·장기 개선 방향 → 기술 스택 → 요약

---

## Notes

- 배경 마스킹(`--mask-bg`): 기존 선별기의 파란 배경에 맞춰 튜닝된 HSV threshold.
  새 촬영 환경에선 thresholds 재조정 필요. Phase 1에선 사용 안 함.
- 라벨 CSV 인코딩: 황금향 `labeling.csv` 는 날짜에 따라 CP949/UTF-8 혼재 —
  `make_labels.py` 에서 양쪽 모두 시도하는 fallback 처리.
- 본 저장소는 실험 기록용이며, 실제 선별기 배포 코드는 별도 리포(ROS1 기반, 비공개).
