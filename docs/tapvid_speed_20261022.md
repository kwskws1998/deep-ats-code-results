# TAP-Vid 4090 실행 속도 개선 — 2026-10-22

기존 실행 폴더를 보존하고 별도 폴더에서 사용한다. 모델 구조·파라미터·학습 데이터·20,000 update·update당 8 query·학습률·FP32를 유지하고 실행 방식만 바꾼다. 4090 실측 속도/VRAM은 아직 확인하지 않았다.

## 구현

| 모듈 | 기능 |
|---|---|
| `tap_features.py` | 한 microbatch의 실제 관측 프레임 union을 encoder chunk로 처리. 같은 프레임을 query마다 다시 전송·인코딩하지 않음 |
| `tap_fast_model.py` | 기존 state_dict와 호환되는 실행 backend. query별 descriptor·belief·decoder는 독립. 공간 좌표 grid도 재사용 |
| `tap_fast_math.py` | SPD 2×2 solve/NLL을 미분 가능한 tensor 연산으로 계산. 프레임별 linalg 오류 검사에 따른 CPU 동기화 제거. 기존 NLL jitter 1e-7 유지 |
| `tap_fast_rollout.py` | 동일 query clock의 추적을 묶어서 수행. uniform/random은 관측 union을 일괄 인코딩. adaptive ATS는 현재 관측 프레임만 인코딩하고 필요한 상태를 결정 시점마다 한 번에 CPU로 복사 |
| `tap_fork.py` | 별도 run으로 기존 last.pt의 weight·Adam state·RNG·학습 step·LR schedule을 이어받고 변경 이력을 기록 |

학습 loss 로그는 logging interval 끝에서 CPU로 복사한다. 매 프레임 반복하던 label/visibility/elapsed tensor 전송을 묶었다. `resources.json`에는 학습·로딩·validation 시간과 backend/batch/chunk를 기록한다. CSV는 과거 로그에 없는 새 열도 보존하므로 legacy history를 이어붙일 수 있다.

관측 횟수는 기존처럼 image-query 쌍마다 청구한다. encoder 공유가 관측 예산을 공짜로 늘리지는 않는다. query가 관측하지 않은 프레임은 그 query의 상태·손실에 들어가지 않는다. 인코더는 GroupNorm을 사용하고 프레임 간 통계 혼합이 없다. 정적 schedule의 선계산 결과도 해당 관측 시점에만 사용한다. 이는 저장된 영상에 대한 batch 실행 시간이며, 실시간 스트리밍 지연/FPS를 측정한 것이 아니다.

비양정치 covariance를 clipping해서 정상으로 위장하지 않는다. 잘못된 SPD pivot은 nonfinite가 되어 학습 update/rollout 경계에서 실패한다. 모든 프레임마다 CPU로 오류를 확인하는 방식은 제거했다. 부동소수점 연산 순서가 달라져 기존 실행과 bitwise 동일하다고 주장하지 않는다.

## 기존 서버 환경 그대로 사용

ZIP을 `/workspace/tapvid_4090_fast_20261022.zip`에 올린 뒤 새 디렉터리에 푼다. 진행 중인 원본 checkout에 덮어쓰면 원본 실행의 source 검증이 실패할 수 있다.

```console
/workspace/deep-ats-code-results/.venv4090/bin/python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
folder = Path('/workspace/deep-ats-fast-v3')
folder.mkdir(exist_ok=False)
with ZipFile('/workspace/tapvid_4090_fast_20261022.zip') as archive:
    archive.extractall(folder)
print(folder)
PY
```

추가 패키지는 필요하지 않다. 모든 명령은 기존 `.venv4090/bin/python`을 사용한다. 기존 GPU 학습과 새 GPU benchmark/학습을 동시에 실행하면 시간 측정과 자원 사용이 서로 영향을 준다.

## 먼저 4090에서 짧은 benchmark

동일한 학습 query·mask·초기 weight·AdamW 설정으로 원본 sequential, 일반 microbatch, encoder 공유 backend를 측정한다. 각 경로 warmup 1 update 뒤 3 update를 측정하고 full-observation validation도 따로 측정한다. 아래 benchmark는 학습 데이터만 사용하며 final-test 파일을 읽지 않는다.

```console
/workspace/deep-ats-code-results/.venv4090/bin/python /workspace/deep-ats-fast-v3/scripts/diagnose_tapvid.py benchmark --data-manifest /workspace/deep-ats-code-results/data/external/tapvid_4090_s128/prepared256_t60/manifest.json --output-root /workspace/deep-ats-fast-v3/outputs/benchmark4090 --batch-size 8 --encoder-chunk-size 8 --max-frames 60 --feature-dim 64 --hidden-size 96 --device cuda --threads 4 --repeats 3
```

`benchmark.csv`의 seconds/update, encoder 호출 수·인코딩 이미지 수·peak VRAM과 `validation_benchmark.csv`를 확인한다. 논리적인 관측 청구량과 실제 CNN 이미지 처리량은 별도로 저장한다. 짧은 probe의 속도는 전체 학습 완료 시간 보장이 아니다. OOM이 나면 먼저 `--batch-size 4 --encoder-chunk-size 4`로 benchmark한다. 실제 학습도 같은 설정을 쓸 때만 해당 측정을 적용한다.

## 새 학습

새 fast profile은 실제 query microbatch 8, encoder chunk 8, validation query batch 8, CPU clip cache 128을 사용한다. 데이터 다운로드·전처리를 반복하지 않도록 원본 경로를 명시한다. 기존 프로파일과 결과 폴더는 유지한다.

```console
/workspace/deep-ats-code-results/.venv4090/bin/python /workspace/deep-ats-fast-v3/scripts/run_tapvid_4090.py all --config /workspace/deep-ats-fast-v3/configs/tapvid_4090_fast.json --raw-root /workspace/deep-ats-code-results/data/external/tapvid_4090_s128 --prepared-root /workspace/deep-ats-code-results/data/external/tapvid_4090_s128/prepared256_t60
```

## 지금 학습한 checkpoint에서 이어가기

원본 프로세스를 완료하거나 정지한 뒤 실행한다. `last.pt`가 마지막으로 저장된 validation step부터 이어받는다. 기존 17,500 update checkpoint라면 17,501부터 진행하며 처음부터 학습하지 않는다. 원본 run을 수정하지 않으며, 새 결과는 `/workspace/deep-ats-fast-v3/outputs/tapvid_4090_s128_fast_v3`에 저장한다.

```console
/workspace/deep-ats-code-results/.venv4090/bin/python /workspace/deep-ats-fast-v3/scripts/run_tapvid_4090.py all --config /workspace/deep-ats-fast-v3/configs/tapvid_4090_fast.json --raw-root /workspace/deep-ats-code-results/data/external/tapvid_4090_s128 --prepared-root /workspace/deep-ats-code-results/data/external/tapvid_4090_s128/prepared256_t60 --fork-from /workspace/deep-ats-code-results/outputs/tapvid_4090_s128_v1/seed17/train/last.pt
```

새 run이 중단되면 같은 명령에서 `--fork-from ...`를 제거하고 `--resume`을 붙인다. 새 run에서 다시 `--fork-from`을 실행해 덮어쓰는 동작은 거부한다. 원본과 데이터 manifest·모델 operator source·과학적 학습 설정·PyTorch/NumPy 버전이 맞는지 검사한다. legacy best/last가 어긋나면 임의로 복구하지 않고 실패한다.

`training_fork.json`에 부모 checkpoint SHA256·source/config·완료 step·실행 변경을 남긴다. 이것은 실행 backend를 변경한 continuation이며 원본과 bitwise 동일한 exact resume가 아니다. 원본 데이터나 학습 조건을 바꾼 채 이어가는 방식은 허용하지 않는다.

## 로컬 검증 범위

실제 4090 실행과 GPU 속도 향상은 이 맥북에서 검증할 수 없다. CPU의 수치·gradient·관측 schedule·미래 프레임 누출 방지·fault recovery·legacy fork와 짧은 실데이터 실행을 검증한다. 세부 실행 결과는 검증 후 아래에 기록한다.

### 확인된 결과

- 전체 테스트: **235 passed, CUDA hardware test 3 skipped**.
- 실제 Kubric 학습 데이터, 256×256, 16프레임, feature 64/hidden 96, query 8, CPU 4 threads, warmup 1회 뒤 3회 평균:

| 실행 | 학습 seconds/update ↓ | full-read validation seconds ↓ |
|---|---:|---:|
| 기존 sequential | 3.3189 | 2.7150 |
| 일반 microbatch | 2.8305 | 별도 측정하지 않음 |
| encoder 공유 | **1.3800** | **0.8562** |

CPU에서 학습 약 **2.41배**, validation 약 **3.17배** 빨랐다. 같은 학습 update의 인코딩 이미지 수는 80→16, encoder 호출은 80→2로 줄었고 관측 청구량은 80으로 같다. 이 결과를 4090의 속도나 60프레임 전체 학습 시간으로 외삽하지 않는다.

- 같은 실제 학습 source의 연속 window를 이어 만든 60프레임·4 query, feature 16/hidden 24 CPU parity probe: 모든 손실 항과 parameter gradient 비교 통과. 최대 손실 절대 차이 1.91e-6, 최대 parameter-gradient 절대 차이 1.49e-5.
- 모든 관측 정책에서 delayed query·관측 mask·예측·Q/R 통계·정책 결정 비교 통과. 정적 schedule 선계산의 미래 프레임이 이전 예측에 영향을 주지 않는 테스트 통과.
- 공유 경로가 동기화형 `torch.linalg.cholesky/solve`를 호출하면 실패하게 한 테스트 통과.
- 새 backend 내부의 중단 후 exact resume는 CPU에서 최종 weight와 history가 bitwise 동일했다.
- 예전 로컬 CPU 실행에서 저장한 실제 legacy checkpoint를 별도 run으로 읽어 후처리/선택까지 완료했고 원본은 보존했다. 이는 서버의 17,500 step checkpoint를 실행한 결과가 아니다.
- benchmark·60프레임 parity·legacy fork artifact의 현재 source 및 저장 파일 hash 검사 모두 통과. `outputs/tapvid_speed_verified_20261022/verification.json`에 기록했다.
