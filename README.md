# Random Selection Contrastive Learning

> SimCLR의 large batch 전략에는 보이지 않는 비용이 있었습니다.

이 프로젝트는 contrastive self-supervised learning에서 batch size를 키울 때 함께 증가하는 **false negative pressure**를 줄이기 위한 연구형 구현입니다. 2022 KCC 논문 `자기지도학습을 위한 랜덤추출 기반 손실함수`의 문제의식을 바탕으로, SimCLR의 full in-batch negative denominator를 그대로 쓰지 않고 anchor별 candidate negative 중 random selected subset만 loss에 노출하는 RSCL(Random Selection Contrastive Learning)을 구현했습니다.

포트폴리오에서 보여주고 싶은 핵심은 단순히 SimCLR을 재현했다는 점이 아닙니다. 연구 질문을 세우고, loss를 코드로 구현하고, false negative exposure라는 분석 지표를 만들고, linear evaluation 결과까지 연결해 해석했다는 점입니다.

## Portfolio Snapshot

| Point | Description |
| --- | --- |
| Problem | SimCLR은 batch size를 키워 negative 수를 늘리지만, self-supervised setting에서는 같은 class sample도 negative로 밀릴 수 있습니다. |
| Idea | positive pair는 유지하고, candidate negative 전체가 아니라 random selected subset만 denominator에 넣습니다. |
| Method | InfoNCE 구조 안에서 selected negative mass만 사용하는 RSCL objective를 PyTorch로 구현했습니다. |
| Evidence | CIFAR label을 사후적으로 사용해 anchor별 false negative exposure를 계산했습니다. |
| Result | CIFAR10 batch 512에서 RSCL이 SimCLR보다 Top-1 `+1.17%p`, CIFAR100 batch 256에서 Top-1 `+0.72%p` 높았습니다. |
| Interpretation | RSCL의 일방적 우위가 아니라, false negative exposure 감소와 informative negative 감소 사이의 trade-off를 확인한 실험입니다. |

## Why This Experiment

Contrastive learning은 anchor와 positive view는 가깝게, batch 안의 다른 sample은 멀게 만들며 representation을 학습합니다. 충분한 negative가 필요하기 때문에 SimCLR은 batch size를 키워 in-batch negative 수를 확보합니다.

하지만 self-supervised setting에서는 label을 쓰지 않습니다. 따라서 anchor와 실제로 같은 class에 속한 이미지도 단지 다른 이미지라는 이유로 negative가 됩니다. batch size가 커질수록 true negative도 늘어나지만, 같은 class sample을 서로 밀어내는 false negative pressure도 함께 증가합니다.

이 프로젝트의 질문은 다음입니다.

```text
더 많은 negative가 항상 좋은가?
candidate negative 전체를 매번 denominator에 넣지 않고,
random selected subset만 사용하면 false negative pressure를 줄일 수 있지 않은가?
```

## What I Implemented

이 저장소는 SimCLR baseline과 RSCL을 같은 코드 구조 안에서 비교할 수 있도록 구성했습니다.

- `simclr/`: full in-batch negative를 사용하는 SimCLR baseline
- `rscl/`: selected negative subset을 사용하는 RSCL
- `common/`: dataset, optimizer, checkpoint, metric, model loading 공용 유틸
- `linear_eval.py`: pretrained encoder를 frozen한 뒤 linear classifier로 평가
- `false_negative_exposure.py`: CIFAR label을 사후적으로 사용한 false negative exposure 분석
- `train_venv.sh`: project-local virtual environment setup and execution helper

실험 설정은 portfolio page와 동일하게 정리했습니다.

| Setting | Value |
| --- | --- |
| Backbone | ResNet-50 |
| Projection dim | 128 |
| Optimizer | LARS |
| Pretrain epochs | 500 |
| Warmup epochs | 10 |
| Learning rate | 0.5 |
| Temperature | 1.0 |
| RSCL k | `batch_size // 4` |
| Datasets | CIFAR10, CIFAR100 |

## RSCL Objective

원본 batch size를 `B`라고 하면 두 개의 augmented view를 통해 총 `2B`개의 projection vector가 만들어집니다. anchor `i`의 positive pair index를 `p(i)`라고 두면 candidate negative set은 다음과 같습니다.

```text
C_i = { j | j != i, j != p(i) }
M = |C_i| = 2B - 2
```

SimCLR은 `C_i` 전체를 denominator에 넣습니다. RSCL은 `C_i`에서 `k`개의 selected negative만 사용합니다.

```text
S_i subset C_i
|S_i| = k
```

구현에서 사용하는 loss는 InfoNCE 구조를 유지하면서 selected negative mass만 denominator에 넣습니다.

```text
l_i =
  -log
  exp(s_i,p(i))
  /
  [ exp(s_i,p(i)) + lambda * sum_{j in S_i} exp(s_i,j) ]
```

현재 구현에서는 `k = batch_size // 4`를 사용합니다. 따라서 batch 256, 512, 1024에서는 각각 k=64, k=128, k=256이 적용됩니다.

## Analysis Evidence

학습에는 label을 사용하지 않지만, 분석 단계에서는 CIFAR train split에서 100개 batch를 샘플링하고 label을 사후적으로 참조했습니다. 목적은 anchor별 candidate negative 안에 같은 label sample이 평균 몇 개 섞이는지 계산하는 것입니다.

중요한 점은 RSCL이 같은-label 비율 자체를 바꾸는 방법이 아니라는 것입니다. RSCL은 candidate negative 전체 중 loss denominator에 실제로 들어가는 수를 제한해, 반복적으로 노출되는 false negative 개수를 줄입니다.

| Dataset | Batch | Candidates | SimCLR FN | FN Ratio | k=64 | k=128 | k=256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR10 | 256 | 510 | 51.20 | 10.04% | **6.42** | 12.85 | 25.70 |
| CIFAR10 | 512 | 1022 | 102.00 | 9.98% | 6.39 | **12.78** | 25.56 |
| CIFAR10 | 1024 | 2046 | 204.30 | 9.99% | 6.39 | 12.78 | **25.57** |
| CIFAR100 | 256 | 510 | 5.10 | 1.00% | **0.64** | 1.28 | 2.55 |
| CIFAR100 | 512 | 1022 | 10.20 | 0.99% | 0.64 | **1.27** | 2.54 |
| CIFAR100 | 1024 | 2046 | 20.40 | 1.00% | 0.64 | 1.28 | **2.56** |

표에서 `SimCLR FN`은 full in-batch negative 후보를 모두 사용할 때 anchor 하나가 마주치는 expected false negative 수입니다. `k=64/128/256`은 RSCL selected negative만 denominator에 넣었을 때 실제로 노출되는 expected false negative 수입니다. 굵은 값은 현재 구현 조건인 `k=batch_size//4`입니다.

CIFAR10에서는 batch size가 커질수록 SimCLR FN이 `51.2 -> 102.0 -> 204.3`으로 증가했습니다. CIFAR10 batch 1024에서 현재 설정인 k=256을 적용하면 loss에 들어가는 expected false negative는 `25.57`개로 줄어, full SimCLR 대비 `12.51%`만 denominator에 노출됩니다.

CIFAR100은 class 수가 많아 같은 batch size에서도 false negative 절대량이 작습니다. 그러나 batch 1024 기준 SimCLR은 anchor당 평균 `20.4`개의 false negative 후보를 마주치고, 현재 RSCL 설정은 `2.56`개만 loss에 노출합니다.

## Linear Evaluation Result

현재 `output/` 폴더에 존재하는 best-result JSON을 기준으로 linear evaluation 결과를 정리했습니다. cell 값은 `Top-1 / Top-5` accuracy입니다.

### CIFAR10

| Model | Batch 256 | Batch 512 | Batch 1024 |
| --- | ---: | ---: | ---: |
| SimCLR | 89.06 / 99.73 | 91.37 / 99.76 | **93.07 / 99.82** |
| RSCL | **89.13 / 99.75** | **92.54 / 99.75** | 91.61 / 99.76 |

### CIFAR100

| Model | Batch 256 | Batch 512 | Batch 1024 |
| --- | ---: | ---: | ---: |
| SimCLR | 59.01 / 85.81 | **62.86 / 88.46** | **70.81 / 92.75** |
| RSCL | **59.73 / 87.26** | 60.44 / 86.94 | - |

CIFAR10 batch 256에서는 RSCL이 SimCLR보다 Top-1 `+0.07%p`, batch 512에서는 `+1.17%p` 높았습니다. 반대로 CIFAR10 batch 1024에서는 SimCLR이 `+1.46%p` 높았습니다.

CIFAR100에서는 batch 256에서 RSCL이 Top-1 `+0.72%p`, Top-5 `+1.45%p` 높았습니다. batch 512에서는 SimCLR이 Top-1 `+2.42%p`, Top-5 `+1.52%p` 높았습니다. `output/rscl/cifar100_batch_1024_best.json`은 현재 output 폴더에 없으므로 해당 조건은 missing으로 처리했습니다.

이 결과는 RSCL이 언제나 SimCLR을 이긴다는 주장이 아닙니다. 더 정확한 결론은 selected negative objective가 false negative exposure를 크게 낮추며, 일부 조건에서는 representation 품질을 유지하거나 개선하지만, dataset과 batch size에 따라 informative negative까지 줄어드는 trade-off가 존재한다는 것입니다.

## Repository Structure

```text
common/
  checkpoints.py
  datasets.py
  false_negative.py
  json_utils.py
  metrics.py
  model_loader.py
  optimizers.py
  training_control.py
  training_outputs.py
simclr/
  config.py
  loss.py
  model.py
  optimizer.py
  train.py
rscl/
  config.py
  loss.py
  model.py
  optimizer.py
  train.py
scripts/
  rscl_smoke_hparam_search.py
linear_eval.py
false_negative_exposure.py
train_venv.sh
```

## Current Output Archive

현재 output archive에는 11개의 best-result JSON과 2개의 false-negative analysis JSON이 있습니다.

```text
output/analysis/cifar10_false_negative_exposure.json
output/analysis/cifar100_false_negative_exposure.json
output/simclr/cifar10_batch_256_best.json
output/simclr/cifar10_batch_512_best.json
output/simclr/cifar10_batch_1024_best.json
output/simclr/cifar100_batch_256_best.json
output/simclr/cifar100_batch_512_best.json
output/simclr/cifar100_batch_1024_best.json
output/rscl/cifar10_batch_256_best.json
output/rscl/cifar10_batch_512_best.json
output/rscl/cifar10_batch_1024_best.json
output/rscl/cifar100_batch_256_best.json
output/rscl/cifar100_batch_512_best.json
```

각 best-result JSON은 `best`, `checkpoint_epoch`, `history`를 포함합니다. 현재 결과의 `history` 길이는 모두 100이며, portfolio table은 이 JSON 값을 기준으로 작성했습니다.

## Reproduction

프로젝트 로컬 venv는 저장소 안의 `.venv`에 생성됩니다.

```bash
./train_venv.sh setup
```

환경 정보 확인:

```bash
./train_venv.sh env-info
```

RSCL pretraining 예시:

```bash
./.venv/bin/python -B rscl/train.py \
  --dataset cifar10 \
  --epochs 500 \
  --batch-size 256 \
  --output-dir rscl/pretrained \
  --learning-rate 0.5 \
  --temperature 1.0 \
  --weight-decay 1e-6 \
  --warmup-epochs 10 \
  --num-workers 4 \
  --device cuda \
  --amp \
  --suppress-external-progress
```

SimCLR baseline pretraining 예시:

```bash
./.venv/bin/python -B simclr/train.py \
  --dataset cifar10 \
  --epochs 500 \
  --batch-size 256 \
  --output-dir simclr/pretrained \
  --learning-rate 0.5 \
  --temperature 1.0 \
  --weight-decay 1e-6 \
  --warmup-epochs 10 \
  --num-workers 4 \
  --device cuda \
  --amp \
  --suppress-external-progress
```

Linear evaluation 예시:

```bash
./.venv/bin/python linear_eval.py \
  --model simclr \
  --checkpoint simclr/pretrained/cifar10_batch_256_best.pt \
  --dataset cifar10 \
  --batch-size 512 \
  --epochs 100
```

False negative exposure analysis 예시:

```bash
./.venv/bin/python false_negative_exposure.py \
  --dataset cifar10 \
  --batch-sizes 256 512 1024 \
  --rscl-k 64 128 256 512 \
  --num-batches 100 \
  --output-path output/analysis/cifar10_false_negative_exposure.json
```

## Interview Summary

면접에서는 이 프로젝트를 다음처럼 설명할 수 있습니다.

```text
SimCLR은 large batch로 negative를 늘리지만,
self-supervised setting에서는 같은 class sample도 negative가 되어 false negative pressure가 함께 커집니다.

RSCL은 positive pair는 그대로 유지하고,
candidate negative 전체가 아니라 random selected subset만 denominator에 넣어
loss에 노출되는 false negative 수를 줄이는 실험입니다.

CIFAR10 batch 1024 기준 full SimCLR은 anchor당 평균 204.3개의 false negative 후보를 포함하지만,
현재 RSCL 설정에서는 25.57개만 loss에 노출됩니다.
Linear evaluation에서는 CIFAR10 batch 512에서 +1.17%p 개선을 확인했고,
동시에 일부 조건에서는 SimCLR이 더 높아 negative 수와 false negative pressure 사이의 trade-off도 확인했습니다.
```
