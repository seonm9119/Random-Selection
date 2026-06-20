# Random Selection Contrastive Learning

이 프로젝트는 2022 KCC 논문 `자기지도학습을 위한 랜덤추출 기반 손실함수`의 아이디어를 다시 정리해서 구현하는 연구 코드입니다. 기존 목표는 SimCLR의 in-batch negative 전체를 그대로 사용하는 방식에서 발생하는 false negative 문제를 줄이는 것이었습니다. 이번 정리에서는 이 아이디어를 단순한 custom loss가 아니라 별도의 SSL 모델인 `RSCL(Random Selection Contrastive Learning)`로 다룹니다.

## 연구 방향

SimCLR 계열의 contrastive learning은 anchor와 positive pair를 가깝게 만들고, 같은 batch 안의 나머지 sample을 negative로 밀어냅니다. 하지만 label을 사용하지 않는 self-supervised setting에서는 실제로 같은 semantic class에 속하는 sample도 negative로 취급될 수 있습니다. batch size가 커질수록 negative 수는 늘어나지만, 동시에 false negative가 포함될 가능성도 커집니다.

RSCL의 방향은 다음과 같습니다.

- 전체 in-batch negative를 항상 모두 사용하지 않는다.
- anchor별 negative 후보 집합에서 일부 negative만 랜덤하게 선택한다.
- false negative가 매 step마다 loss denominator에 노출되는 빈도와 영향력을 줄인다.
- encoder/projection 구조는 SimCLR 기준선과 비교 가능하게 유지하되, 모델 이름과 학습 설정은 독립적으로 관리한다.

## 기존 수식의 문제

논문 PDF의 핵심 수식은 다음 의도를 갖고 있었습니다.

```text
2N - 2개의 negative sample 중 c개를 랜덤 추출해서 negative 공간을 다시 구성한다.
```

다만 PDF의 식 (3)은 다음과 같은 형태였습니다.

```text
Random negative =
  sum_{n in N} exp(sim(z_i, z_n) / tau)
  /
  sum_{p in P} exp(sim(z_i, z_p) / tau)
```

이 식은 다음 문제가 있습니다.

- `P`가 self-supervised setting에서 명확히 정의되지 않습니다.
- InfoNCE의 denominator는 같은 스케일의 exponential similarity mass를 더해야 하는데, 위 식은 두 합의 비율입니다.
- `exp(pos) + Random negative`처럼 더하면 positive term과 negative term의 차원이 맞지 않습니다.
- Bernoulli likelihood처럼 전개한 부분은 `Pos`, `Neg`를 count 또는 probability처럼 다루지만, 실제 항은 similarity exponent sum이라 해석이 맞지 않습니다.
- legacy 구현은 positive pair index가 아니라 feature 값 비교로 mask를 만들고 있어 수식과 구현이 모두 불안정했습니다.

따라서 이번 구현에서는 기존 식의 의도만 유지하고, loss는 InfoNCE 구조를 보존하는 형태로 다시 정의합니다.

## RSCL 수식

원본 batch size를 `B`라고 하면 두 개의 augmentation view를 만든 뒤 총 `2B`개의 projection vector를 얻습니다.

```text
z_1, z_2, ..., z_{2B}
```

anchor `i`의 positive pair index를 `pi(i)`라고 둡니다. candidate negative set은 anchor 자신과 positive pair를 제외한 나머지입니다.

```text
C_i = { a | a != i, a != pi(i) }
M = |C_i| = 2B - 2
```

RSCL은 `C_i`에서 `k`개의 negative를 랜덤하게 선택합니다.

```text
S_i ~ Uniform subsets of C_i
|S_i| = k
```

cosine similarity는 다음과 같습니다.

```text
s(i, a) = sim(z_i, z_a)
```

selected negative mass는 다음으로 정의합니다.

```text
RandomNegative_i =
  lambda * sum_{n in S_i} exp(s(i, n) / tau)
```

최종 loss는 InfoNCE의 denominator 구조를 유지합니다.

```text
l_i =
  -log
  exp(s(i, pi(i)) / tau)
  /
  [ exp(s(i, pi(i)) / tau) + RandomNegative_i ]

L_RSCL =
  (1 / 2B) * sum_i l_i
```

여기서 `k = 2B - 2`, `lambda = 1`이면 full in-batch negative를 사용하는 SimCLR/NT-Xent 형태로 돌아갑니다. `k < 2B - 2`이면 RSCL의 핵심인 random selected negative training이 됩니다.

## False Negative 해석

anchor `i`에 대한 실제 false negative 집합을 `F_i`라고 하고, 그 크기를 `f`라고 둡니다.

```text
F_i subset C_i
|F_i| = f
```

전체 후보 negative 수는 `M = 2B - 2`이고 RSCL이 선택하는 negative 수는 `k`입니다.

선택된 negative 안에 포함되는 false negative의 기대 개수는 다음과 같습니다.

```text
E[ |S_i intersect F_i| ] = kf / M
```

false negative가 하나 이상 선택될 확률은 다음과 같습니다.

```text
P(S_i contains at least one false negative)
= 1 - C(M - f, k) / C(M, k)
```

SimCLR은 batch 안에 false negative가 존재하면 항상 denominator에 포함합니다. 반면 RSCL은 `k`를 줄여 false negative가 실제 loss에 노출되는 빈도를 낮춥니다. 이 점이 RSCL을 단순한 loss variation이 아니라 false negative 완화 모델로 설명할 수 있는 핵심입니다.

## 코드 구조

```text
benchmark/
  simclr/
    config.py
    model.py
    loss.py
    optimizer.py
    train.py
  byol/
    config.py
    model.py
    loss.py
    optimizer.py
    train.py
  simsiam/
    config.py
    model.py
    loss.py
    optimizer.py
    train.py
rscl/
  config.py
  model.py
  loss.py
  optimizer.py
  train.py
```

비교군은 다음처럼 사용합니다.

- `SimCLR`: full in-batch negative baseline
- `RSCL`: random selected negative를 사용하는 제안 모델
- `BYOL`: negative sample이 없는 SSL 비교군
- `SimSiam`: negative sample이 없는 SSL 비교군

## RSCL 설정

RSCL의 주요 설정은 `rscl/config.py`에 있습니다.

```text
RANDOM_NEGATIVE_COUNT = 128
NEGATIVE_MASS_SCALE = 1.0
TEMPERATURE = 0.5
LEARNING_RATE = 0.3
WARMUP_EPOCHS = 10
BATCH_SIZE = 1024
BACKBONE_NAME = "resnet18"
```

`RANDOM_NEGATIVE_COUNT`는 anchor별로 선택할 negative 수입니다. `None`으로 바꾸면 가능한 모든 candidate negative를 사용합니다. 이 경우 RSCL loss는 SimCLR의 full negative setting에 가까워집니다.

`NEGATIVE_MASS_SCALE`은 selected negative mass에 곱하는 계수입니다. 기본값은 `1.0`이며, random selection을 통한 negative pressure 감소 효과를 그대로 둡니다.

## 환경 설정

프로젝트 로컬 venv는 repository 안에 생성됩니다.

```bash
./train_venv.sh setup
```

환경 정보 확인:

```bash
./train_venv.sh env-info
```

## 학습 실행

최종 benchmark pretraining은 fixed epoch이 아니라 `max epoch + train loss early stop` 방식으로 실행합니다. SSL pretraining loss는 downstream representation quality와 직접적으로 맞지 않을 수 있으므로, validation loss는 중단 기준이 아니라 상태 감시와 diagnostic best checkpoint 저장에만 사용합니다.

```text
max pretrain epochs: 400
validation split: CIFAR train split에서 5,000장
max batch size: 1024
save every: 100 epochs
validation early stopping: disabled
train loss stop: enabled
train loss stop start: warmup 이후
train loss stop min delta: 0.01
train loss stop patience: 5 epochs
```

warmup 이후에도 train loss가 충분히 내려가지 않으면 해당 learning rate는 부적절한 것으로 보고 중단한 뒤 learning rate를 조정해서 다시 실행합니다. `400` epoch은 끝까지 무조건 돌린다는 뜻이 아니라 최대로 허용하는 상한입니다.

현재 벤치마크 사전학습 계획은 CIFAR100과 CIFAR10 모두에 대해 동일하게 실행합니다.

```text
datasets: cifar100, cifar10
SimCLR: 1024, 512, 256, 128
BYOL: 1024
SimSiam: 1024
RSCL: benchmark 실행 목록에서는 제외, 같은 train-loss-stop policy 사용
```

따라서 benchmark launcher는 총 12개 run을 순차 실행합니다.

벤치마크 전체 순차 학습:

```bash
./.venv/bin/python train_benchmarks.py
```

RSCL 학습:

```bash
./.venv/bin/python rscl/train.py
```

SimCLR baseline 학습:

```bash
./.venv/bin/python benchmark/simclr/train.py
```

BYOL 학습:

```bash
./.venv/bin/python benchmark/byol/train.py
```

SimSiam 학습:

```bash
./.venv/bin/python benchmark/simsiam/train.py
```

비교 모델의 결과는 `benchmark/<model_name>/pretrained/` 바로 아래에 저장되고, 제안 모델 RSCL의 결과는 `rscl/pretrained/` 아래에 저장됩니다.

학습 스크립트는 epoch checkpoint와 JSONL train log를 저장하지 않고, 같은 basename의 best checkpoint와 config JSON만 저장합니다. Benchmark launcher는 같은 basename의 console log를 각 모델의 `pretrained/` 폴더 바로 아래에 저장합니다. 예를 들어 CIFAR100 batch 1024는 각 모델 폴더에 다음 파일만 남깁니다.

```text
benchmark/simclr/pretrained/cifar100_batch_1024_best.pt
benchmark/simclr/pretrained/cifar100_batch_1024_best.json
benchmark/simclr/pretrained/cifar100_batch_1024_best.log

benchmark/byol/pretrained/cifar100_batch_1024_best.pt
benchmark/byol/pretrained/cifar100_batch_1024_best.json
benchmark/byol/pretrained/cifar100_batch_1024_best.log

benchmark/simsiam/pretrained/cifar100_batch_1024_best.pt
benchmark/simsiam/pretrained/cifar100_batch_1024_best.json
benchmark/simsiam/pretrained/cifar100_batch_1024_best.log

rscl/pretrained/cifar100_batch_1024_best.pt
rscl/pretrained/cifar100_batch_1024_best.json
rscl/pretrained/cifar100_batch_1024_best.log
```

2026-06-20에 중지한 `final_cifar100_batch_1024` run은 epoch 232까지 진행됐지만 train loss 하강이 약했습니다. 이후 SimCLR loss를 두 방향 평균 NT-Xent로 수정했고, SimCLR base learning rate를 `1.0`, warmup을 `10` epoch으로 조정했습니다. RSCL의 learning rate는 아직 별도 탐색 전이므로 `0.3`으로 남겨둡니다. 산출물은 dataset과 batch size 기준의 고정 파일명으로 덮어씁니다.

SimCLR CIFAR100 batch 1024 기준 learning rate 탐색 결과는 다음과 같습니다. 모든 값은 warmup 10 epoch 이후 train loss stopper로 확인했습니다.

```text
base lr 0.030: best val 6.5067, train drop 0.1362, stopped
base lr 0.075: best val 6.4055, train drop 0.1554, stopped
base lr 0.150: best val 6.2953, train drop 0.2027, not stopped
base lr 0.300: best val 6.2252, train drop 0.1908, not stopped
base lr 1.000: best val 6.1649, train drop 0.1784, stopped
```

현재 SimCLR 추천값은 `LEARNING_RATE = 1.0`입니다. batch 1024에서는 linear scaling으로 실제 peak learning rate가 `4.0`이 되며, cosine schedule과 10 epoch warmup을 함께 사용합니다. `1.0`은 warmup 중 validation loss가 흔들리지만 NaN/OOM 없이 후반 cosine decay 구간에서 가장 낮은 validation loss를 기록했습니다.

콘솔 로그 파일은 학습 스크립트의 epoch summary를 저장합니다.

## 공용 평가 및 분석 파일

비교 모델 학습 코드는 `benchmark/<model>/train.py`에 두고, 제안 모델 학습 코드는 `rscl/train.py`에 둡니다. 학습이 끝난 checkpoint를 읽어서 결과를 뽑는 코드는 공용 파일로 분리합니다.

```text
common/
  checkpoints.py
  datasets.py
  false_negative.py
  json_utils.py
  metrics.py
  model_loader.py
  optimizers.py
linear_eval.py
false_negative_exposure.py
```

`common/model_loader.py`는 `simclr`, `rscl`, `byol`, `simsiam` checkpoint를 같은 방식으로 로드하고 encoder feature extractor를 만들어줍니다. 따라서 평가 코드는 모델별 내부 구조를 직접 알 필요가 없습니다.

Linear evaluation 실행 예시는 다음과 같습니다.

```bash
./.venv/bin/python linear_eval.py \
  --model simclr \
  --checkpoint benchmark/simclr/pretrained/cifar100_batch_1024_best.pt \
  --dataset cifar100 \
  --batch-size 512 \
  --epochs 100
```

False negative exposure 분석 실행 예시는 다음과 같습니다.

```bash
./.venv/bin/python false_negative_exposure.py \
  --dataset cifar100 \
  --batch-sizes 64 128 256 512 \
  --rscl-k 64 128 \
  --num-batches 100
```

결과 파일은 기본적으로 `results/linear_eval/`, `results/analysis/` 아래에 JSON으로 저장됩니다.

## 실험 결과 설계

이 연구의 메인은 단순한 custom loss 실험이 아니라 `False Negative 문제를 줄이기 위한 SimCLR 개선 연구`입니다. 따라서 결과도 accuracy 하나만 비교하는 방식보다, batch size가 커질 때 SimCLR과 RSCL이 어떻게 다르게 반응하는지 보여주는 방향으로 설계합니다.

핵심 가설은 다음과 같습니다.

```text
SimCLR은 batch size가 커질수록 더 많은 in-batch negative를 사용해서 성능이 좋아질 수 있다.
하지만 batch가 커질수록 false negative가 denominator에 포함되는 절대량도 증가한다.

RSCL은 anchor별 selected negative 수를 제어하므로,
batch size가 커져도 false negative exposure가 크게 증가하지 않는다.
따라서 RSCL은 batch size 변화에 덜 민감하고 더 안정적인 downstream accuracy를 보일 수 있다.
```

정확한 표현은 다음처럼 잡습니다.

```text
SimCLR 성능 향상: 더 많은 in-batch negative 덕분
SimCLR 문제: batch가 커질수록 false negative도 더 많이 loss에 노출됨
RSCL 개선: selected negative만 사용해 false negative exposure를 제어함
```

### 1. Batch Size별 Downstream 성능

먼저 SimCLR과 RSCL을 동일한 조건에서 batch size별로 사전학습한 뒤 linear evaluation을 수행합니다.

필수 지표:

- Linear evaluation Top-1 accuracy
- Linear evaluation Top-5 accuracy
- 가능하면 kNN accuracy를 보조 지표로 추가

기본 결과표 형식:

```text
CIFAR-100, ResNet-18, 400 pretrain epochs

Model       Batch 128   Batch 256   Batch 512   Batch 1024
SimCLR
RSCL-k64
RSCL-k128
```

최종 보고용으로는 CIFAR-10과 CIFAR-100을 모두 사용합니다.

```text
Dataset: CIFAR-10, CIFAR-100
Backbone: modified ResNet-18
Epochs: max 400 with train-loss-stop pretraining
Batch sizes: 128, 256, 512, 1024
Models: SimCLR, RSCL-k64, RSCL-k128
Seeds: 최소 3개
Metrics: Linear Top-1, Linear Top-5, false negative exposure, batch sensitivity
```

### 2. Batch Sensitivity 지표

batch size별 accuracy 표만으로는 RSCL이 안정적인지 강하게 주장하기 어렵습니다. 따라서 batch 변화에 대한 민감도를 숫자로 요약합니다.

추천 지표:

```text
range = max(accuracy) - min(accuracy)
std = std(accuracy across batch sizes)
slope = accuracy change per log2(batch size)
```

기대하는 패턴:

```text
SimCLR: batch size에 따라 성능 변화가 큼
RSCL: batch size가 바뀌어도 성능 변화가 작음
RSCL: 작은 batch 또는 중간 batch에서 SimCLR보다 우수하거나 비슷함
```

추천 그래프:

```text
x축: batch size
y축: Top-1 accuracy
line: SimCLR, RSCL-k64, RSCL-k128
```

### 3. False Negative Exposure 분석

학습 중에는 label을 쓰지 않지만, 분석 단계에서는 CIFAR label을 사용해서 false negative exposure를 측정할 수 있습니다. 이 지표는 RSCL의 연구 방향을 가장 직접적으로 보여줍니다.

anchor `i`에 대해 다음 집합을 정의합니다.

```text
C_i = candidate negatives
F_i = same-label samples inside C_i
S_i = RSCL selected negatives
```

측정할 값:

```text
SimCLR false negative count = |F_i|
RSCL false negative count = |S_i intersect F_i|
RSCL false negative exposure ratio = |S_i intersect F_i| / |S_i|
RSCL exposure fraction vs SimCLR = |S_i| / |C_i|
```

batch size별 기대 패턴:

```text
SimCLR: batch가 커질수록 |F_i| 증가
RSCL-k fixed: |S_i intersect F_i|는 비교적 일정
```

주의할 점은, RSCL이 negative를 균등 랜덤 선택한다면 같은-label negative의 비율 자체는 기대값으로 크게 바뀌지 않을 수 있다는 것입니다. 핵심은 `비율 감소`가 아니라, loss denominator에 실제로 들어가는 false negative의 `개수`와 `노출량`을 줄이는 것입니다.

추천 그래프:

```text
x축: batch size
y축: average false negative count per anchor
line: SimCLR, RSCL-k64, RSCL-k128
```

이 그래프는 RSCL이 단순히 loss 값을 바꾼 것이 아니라 false negative 노출량을 실제로 제어한다는 주장을 뒷받침합니다.

### 4. Representation 품질 분석

RSCL이 negative를 덜 사용해서 쉬운 loss만 만든 것이라는 반론을 막으려면 representation 품질 지표를 함께 확인하는 것이 좋습니다.

추천 지표:

- positive pair similarity
- same-class similarity
- different-class similarity
- alignment
- uniformity

기대하는 해석:

```text
RSCL은 false negative exposure를 낮추면서도 collapse하지 않는다.
RSCL은 같은 class sample을 과하게 밀어내는 현상을 줄인다.
RSCL은 downstream linear evaluation에서 안정적인 class separation을 유지한다.
```

### 5. 최종 주장 형태

실험 결과가 목표대로 나오면 결론 문장은 다음 형태로 정리합니다.

```text
RSCL은 SimCLR처럼 large batch에 강하게 의존하지 않으면서도,
false negative exposure를 낮추고,
batch size 변화에 대해 더 안정적인 downstream accuracy를 보인다.
```

## 실험 메모

논문 PDF의 CIFAR 설정은 다음 기준으로 다시 반영했습니다.

- dataset: CIFAR-10, CIFAR-100
- image size: 32x32
- encoder: modified ResNet-18
- first convolution: 3x3, stride 1
- max pooling 제거
- projection dimension: 128
- augmentation: Inception crop, color distortion strength 0.5
- optimizer: LARS
- scheduler: cosine annealing
- learning rate: paper setting 1.5, corrected SimCLR default 1.0 with linear batch scaling
- warmup epochs: 10
- temperature: 0.5
- batch size: 256, 512
- pretrain epochs: 100, 500

최종 benchmark 실행은 RTX 3060에서의 실행 가능성과 결과 신뢰도를 함께 고려해서 400 epoch 고정 학습을 기본값으로 둡니다. 논문 appendix의 500 epoch 설정을 완전히 맞추는 추가 run이 필요하면 `RS_BENCHMARK_EPOCHS=500`으로 launcher를 실행합니다.

현재 구현은 ResNet-18과 ResNet-50 모두 선택할 수 있게 구성했습니다.
