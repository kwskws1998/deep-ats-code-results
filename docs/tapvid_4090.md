# RTX 4090 실험 실행

2026-10-22 추가 구현과 코드 리뷰는 [별도 진단 가이드](tapvid_review_20261022.md)를 참고한다. 진행 중인 서버 checkout에 새 소스를 덮어쓰지 않고 별도 폴더에서 checkpoint를 진단한다. 실제 query microbatch 4 설정은 `configs/tapvid_4090_batch4.json`으로 opt-in한다.

우리 ATS-CNN만 학습하고, 외부 비교 모델은 **Track-On-R 하나**를 사용한다. 상대 모델의 학습·추론·가중치 다운로드 없이 논문 Table 1의 DAVIS 결과를 가져온다. 이 문서는 실험 실행 방법이며, 아직 4090 실험이 완료되었다는 보고서가 아니다.

## 환경 설정

Linux x86_64, Python 3.12와 사용 가능한 NVIDIA 드라이버가 있는 RTX 4090 서버에서 실행한다. Ubuntu 24.04와 Python venv를 사용할 수 있는 서버 이미지가 적합하다. 시스템 RAM은 16 GB 이상, 가능하면 32 GB를 준비한다. 128개 Kubric 원본은 대략 수십 GB이며 전처리 데이터가 추가된다. 여유 공간은 안내만 출력하며, 100 GiB 또는 다른 고정 최소 용량으로 실행을 차단하지 않는다. 실제 파일시스템 용량이 소진되면 해당 파일 쓰기 오류가 발생한다. 기존 config의 `minimum_prepare_free_gib` 항목은 무시하며, 공간에 따라 장면 수나 학습 설정을 자동 축소하지 않는다. 처음부터 전체 약 6,000개 장면을 받지는 않는다.

프로젝트 압축을 푼 디렉터리에서 다음을 실행한다. `setup_tapvid_4090.py`는 `.venv4090`를 만들고 CUDA용 PyTorch와 vision 실험 의존성만 설치한다. 기존 인지모델/RL 패키지 전체를 설치할 필요가 없으며, venv 활성화도 필요 없다.

```text
python3.12 scripts/setup_tapvid_4090.py
.venv4090/bin/python scripts/run_tapvid_4090.py plan
.venv4090/bin/python scripts/run_tapvid_4090.py all
```

설치 명령을 먼저 확인하려면 다음을 실행한다. 이 명령은 설치하거나 다운로드하지 않는다.

```text
python3.12 scripts/setup_tapvid_4090.py --dry-run
```

고정한 핵심 버전은 Python 3.12, `torch==2.12.1+cu126`, `numpy==2.2.6`, `Pillow==11.3.0`, `matplotlib==3.10.6`, `pytest==8.4.2`다. CUDA 휠에는 실행에 필요한 사용자 공간 라이브러리가 포함되지만 호스트 NVIDIA 드라이버는 필요하다. CUDA Toolkit을 소스 빌드용으로 따로 설치하는 과정은 포함하지 않는다. 선택한 휠은 [공식 PyTorch 인덱스](https://download.pytorch.org/whl/cu126/torch/)에서 제공한다. 전이 의존성은 설치 시 해결하고, 실제 설치 결과를 `.venv4090/resolved-requirements.txt`에 저장한다. 이것은 모든 전이 의존성까지 고정한 lockfile이라고 주장하지 않는다.

GPU가 연결되지 않았거나 드라이버가 맞지 않으면 설치 마지막 확인 또는 preflight가 명시적으로 실패한다. Mac에서 CUDA 설치·4090 실행을 검증한 것으로 표시하지 않는다.

## 실행 단계

`all`은 아래 단계를 순서대로 수행한다. 단계별 실행도 가능하다.

```text
.venv4090/bin/python scripts/run_tapvid_4090.py prepare
.venv4090/bin/python scripts/run_tapvid_4090.py preflight
.venv4090/bin/python scripts/run_tapvid_4090.py train
.venv4090/bin/python scripts/run_tapvid_4090.py validate
.venv4090/bin/python scripts/run_tapvid_4090.py test
.venv4090/bin/python scripts/run_tapvid_4090.py report
```

`train`도 CUDA preflight를 자동 실행하므로 `all` 실행 시 별도 preflight 명령은 필요 없다. preflight는 실제 학습 clip 하나와 버리는 모델을 사용해 1회 준비 운동 후 3회 query의 forward/backward/optimizer 실행 시간을 측정한다. 60프레임 전부를 읽는 조건에서 peak allocated/reserved VRAM을 기록한다. 단일 clip에서 추정한 시간은 데이터 로딩, validation, 최종 calibration 및 test 시간을 보장하지 않는다. 실제 학습은 각 validation 시점에 학습 step과 validation의 시간을 따로 측정해 남은 소요 시간을 다시 추정한다.

서버가 꺼졌거나 시간 제한으로 deferred되면 같은 설정으로 재개한다. 학습 과정에는 optimizer와 random state를 포함하는 `last.pt`를 저장한다. 마지막 durable checkpoint 이후 진행분은 다시 계산될 수 있다.

```text
.venv4090/bin/python scripts/run_tapvid_4090.py all --resume
```

예상 잔여 학습·validation 시간이 12시간을 넘어서 deferred되었고 더 오래 실행할 계획이면 시간 허용치만 변경할 수 있다.

```text
.venv4090/bin/python scripts/run_tapvid_4090.py all --resume --max-hours 24
```

`--max-hours`는 잔여 시간 추정에 대한 중단 기준이며, 프로세스를 지정된 시각에 강제 종료하는 타이머가 아니다. calibration/test는 이 추정에 포함되지 않는다. Deferred 학습에서는 final test를 실행하지 않고 종료 코드 2를 반환한다. 완료된 학습은 다시 학습하지 않는다. 소스·데이터·학습 설정을 바꾸려면 새 output root와 새 실험을 사용한다. 중단된 validation/test는 `--resume`에서 기존 폴더를 `*_interrupted_시간` 이름으로 보존하고, 같은 frozen checkpoint로 평가 전체를 다시 실행한다. 개별 영상부터 이어서 계산하는 기능은 아니다. 원본 데이터 다운로드는 byte-range 재개를 지원하고, 중단된 전처리는 이 실행기가 소유한 staging 결과만 재생성한다.

처음부터 3개 seed를 실행하려면 다음처럼 선언한다. seed를 test 결과를 보고 고르는 용도로 사용하지 않는다.

```text
.venv4090/bin/python scripts/run_tapvid_4090.py all --seeds 17 29 43 --output-root outputs/tapvid_4090_s128_3seeds
```

## 고정 실험 설정

설정 파일은 `configs/tapvid_4090.json`이다. 아래 값은 4090용으로 지정한 시작점이며, 성능 최적점이나 수렴이 입증된 값이 아니다.

| 항목 | 설정 |
|---|---|
| 입력 / metric 해상도 | 256×256 / 256×256 |
| Kubric 원본 | 고정 revision의 장면 0000–0127 |
| source-level split | 115개 train / 13개 validation, split seed 17 |
| 학습 temporal window | 60프레임, 장면당 2개 |
| CNN feature / GRU hidden | 64 / 96 |
| 학습 업데이트 | 20,000 |
| update당 query | 8개, 하나씩 backward 후 gradient 누적 |
| optimizer | AdamW, weight decay 0.0001, gradient norm clip 5 |
| learning rate | peak 0.0003, 1,000-step linear warmup, cosine decay, 마지막 0.00003 |
| 정밀도 | FP32 텐서, autocast/AMP 사용 안 함 |
| checkpoint 선택 | 매 500 step 및 최초/최종 step의 validation AJ, 전체 관측·uniform |
| 선택용 validation 표본 | 고정 8개 source video × 최대 16개 query, 전체 120프레임 |
| visibility threshold | validation에서만 선택 |
| train read masks | 관측 비율 25/50/75/100%에서 무작위 |
| test | DAVIS 전체 30개 영상·650개 first-visible query |
| test 관측 비율 | 25%, 50%, 100% |
| acquisition 비교 | 동일 모델·가중치의 uniform 대 ATS |
| 외부 상대 모델 | Track-On-R 논문 수치만 사용 |

60프레임 unroll과 Gaussian covariance 계산 때문에 혼합 정밀도를 자동으로 켜지 않았다. GPU 이름, PyTorch/CUDA 버전, TF32 backend 설정, 실측 VRAM과 실행 시간은 preflight에 저장된다. 이 작은 CNN을 24 GB VRAM에 가득 채우는 것이 목표가 아니며, 8개 query는 동시 batch가 아니다. 요청 관측 비율에 남은 프레임 수를 곱한 값을 올림하여 query 프레임을 포함한 정수 관측 횟수를 정한다. 그림의 x축은 요청 budget이고 실제 charged count는 예측 mask와 metric 파일에 저장된다. 데이터 크기, 모델 크기, 학습량은 Track-On-R과 다르다. 128-scene subset으로 전체 synthetic/real-world 학습을 재현한다고 주장하지 않는다.

DAVIS는 gradient fitting, checkpoint 선택, threshold 선택에 사용하지 않는다. 매 step에는 train만 사용한다. 최종 validation/test는 선택 완료된 checkpoint를 고정하고 실행한다. 이 결과를 확인한 후 모델이나 설정을 수정하면 이후 DAVIS 결과는 개발 과정의 반복 평가가 되므로, 논문 최종 주장에는 별도의 미사용 benchmark가 필요할 수 있다.

## 공개 수치 비교 조건

[Track-On-R 논문](https://arxiv.org/html/2603.12217v1)의 Table 1에서 DAVIS의 AJ / within-threshold / OA 원문 수치는 “68.1 80.3 92.5”다. 단위는 %이며 클수록 좋다. 논문에서 확인한 원문 protocol 발췌, 출처, 접근 날짜와 공식 코드 확인 내역은 `references/track_on_r_davis_first.json`에 저장했다. 이번 검색에서 확인한 동일 256×256·query-first 조건의 단일 tracker 중 가장 높은 공개 AJ를 기준으로 선택했으며, 해상도나 strided-query 조건을 섞은 범용 1위 주장으로 확대하지 않는다.

우리 모델의 **100% 관측 결과**만 같은 관측 조건의 공개 reference와 기술적으로 대조한다. 25/50% ATS의 비교 상대는 동일 checkpoint·동일 관측 횟수의 uniform이다. Track-On-R의 부분 관측 성능은 알려져 있지 않으므로 빈칸으로 남기며, 전체 관측 수치를 sparse budget의 matched baseline으로 재사용하지 않는다.

외부 논문은 추가 real-world fine-tuning 및 pretrained DINOv3를 사용한다. [공식 평가 코드](https://github.com/gorkaydemir/track_on/blob/main/evaluation/eval.py)는 joint query와 `support_grid_size=20`도 사용하지만 우리 모델은 독립 query 방식이다. 따라서 benchmark system 결과를 기술적으로 나란히 제시할 수 있어도, 동일 데이터·backbone·compute·query context로 통제한 ATS 효과 비교는 아니다. 공개 aggregate 수치만으로 paired confidence interval, 유의성, 동일 GPU 속도/VRAM 우위를 계산하지 않는다.

## 결과 파일과 검증

기본 결과 위치는 `outputs/tapvid_4090_s128_v1/`다.

| 파일 | 내용 |
|---|---|
| `preflight/cuda_*.json` | 실제 GPU 환경, 학습 backward 메모리와 시간 |
| `seed17/train/model.pt` | validation으로 선택·calibration 완료한 가중치 |
| `seed17/train/last.pt` | 중단 재개용 상태 |
| `seed17/train/training.csv` | validation 학습 곡선 |
| `seed17/train/resources.json` | 실제 학습/validation 시간, GPU peak 메모리 |
| `seed17/validation/` | 고정 validation subset의 별도 평가 |
| `seed17/test/` | 전체 DAVIS 예측, 관측 mask, per-video 수치 |
| `report/comparison.csv` | 우리 실측 결과 + 외부 공개 reference 하나 |
| `report/aggregate.csv` | seed 평균과, 여러 seed이면 표준편차 |
| `report/budget_comparison.png` / `.pdf` | 관측량별 성능과 100%에서만 놓인 외부 reference 점 |
| `report/results.md` | 조건 차이와 출처가 포함된 읽기 쉬운 결과표 |
| `report/audit.json` | 공식 metric 재계산·정확한 관측 예산·전체 query 검증 |

report는 hash를 확인한 공식 TAP-Vid metric으로 모든 예측을 다시 채점한다. 30개 영상·650개 query, 실제 이미지 해상도, first query, query 이전 제외, per-query budget, video macro average와 100%에서 uniform/ATS 예측 일치까지 확인한다. subset을 전체 benchmark 결과인 것처럼 비교하지 못하게 막는다. 외부 수치는 paper provenance로, 우리 수치는 fresh execution으로 구분한다.

```text
.venv4090/bin/python -m pytest -q tests/test_tapvid.py tests/test_tapvid_server.py
.venv4090/bin/python scripts/verify_artifacts.py outputs/tapvid_4090_s128_v1/seed17/train --check-current-source
.venv4090/bin/python scripts/verify_artifacts.py outputs/tapvid_4090_s128_v1/seed17/test --check-current-source
```

서버 코드만 새로 묶으려면 다음을 실행한다. 기존 zip은 덮어쓰지 않는다.

```text
python scripts/package_tapvid_4090.py --output outputs/tapvid_4090_code.zip
```
