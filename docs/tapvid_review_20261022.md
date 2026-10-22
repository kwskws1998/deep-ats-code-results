# TAP-Vid 코드 리뷰와 진단 모듈

검토 범위는 `ats_vision`의 TAP-Vid 데이터·학습·추론·평가·서버 실행 경로와 공통 artifact 저장 코드다. 인지모델과 RL 모델 전체를 별도로 재감사했다는 의미는 아니다. 서버에서 보고된 1,500-update checkpoint는 로컬에 없으므로 그 가중치의 실패 원인을 확정하지 않았다.

## 진행 중인 서버 실험과 분리

새 코드는 **별도 폴더**에 풀어서 사용한다. 실행 중인 `/workspace/deep-ats-code-results`의 `src`, `scripts`, `tests`에 덮어쓰면 원래 실험의 source fingerprint가 바뀌어 완료·재개 검사가 실패한다. 원래 실행을 그대로 유지하고 기존 venv의 Python으로 새 폴더의 진단 스크립트를 실행할 수 있다. 환경을 다시 설치할 필요는 없다.

기존 `model.pt`와 `last.pt`는 수정하지 않는다. 진단은 파일을 한 번 읽은 byte snapshot을 사용하며 원래 파일이 다음 step에 교체되어도 진단 중 가중치는 고정된다. 진단 폴더에 별도 `diagnostic_snapshot.pt`를 저장한다. 진단의 compatibility 예외는 모델 연산 소스 두 파일의 hash가 정확히 같을 때만 허용한다. main 학습 재개·최종 test의 엄격한 source 검사는 유지한다. framework/device 차이는 provenance에 기록하며 GPU와 CPU의 완전 동일 재현을 주장하지 않는다.

## 재현한 문제와 수정

| 우선순위 | 문제 | 영향 | 수정·검증 |
|---|---|---|---|
| P1 | best `model.pt`와 `last.pt`를 따로 저장하는 사이에 중단 | 마지막 durable optimizer/RNG 상태보다 미래의 best 모델 파일을 재개 시 사용할 수 있음 | `last.pt` 안에 best 상태의 독립 복사본을 함께 저장하고 재개 시 복구. 두 파일 저장 사이에 오류를 주입해 uninterrupted 결과와 가중치 일치 확인 |
| P2 | JSON/CSV의 직접 덮어쓰기 | 기록 중 종료 시 manifest·로그가 잘려 재개/모니터링을 방해할 수 있음 | 완성한 임시 파일을 atomic rename으로 게시. 교체 실패 시 이전 JSON 보존 검사. 전원 장애까지의 파일시스템 내구성 보장은 아님 |
| P2 | query 8개를 하나씩 backward | 4090에서 작은 연산을 반복하며 병렬 처리가 제한됨 | `tap_batching.py`에 실제 microbatch 추가. query별 mask·belief·loss 분모 유지. CPU loss·gradient parity 및 unread-query 정보 분리 검사. CUDA 검사는 GPU가 있을 때 실행 |
| P2 | AJ와 직전 update의 total loss 위주 기록 | 위치·가림 실패와 loss 항목별 변화를 구분하기 어려움 | 위치/가림 metric, loss 구성 요소의 logging interval 평균, LR, query 수 기록 |
| P2 | 전체 관측에서도 ATS가 후보별 미래 covariance를 계산 | 선택이 강제되어 있는데 불필요한 CPU 연산 발생 | 남은 프레임 수와 남은 budget이 같으면 바로 다음 프레임 선택. 실제 예측과 budget은 유지 |
| P2 | 100 GiB/20 GiB 고정 디스크 차단 | 충분할 수 있는 디스크에서도 다운로드 전에 종료 | 차단 제거. 여유 용량은 안내만 출력하며 장면 수를 자동 변경하지 않음 |
| P2 | GPU 총용량 20 GiB와 GPU 이름을 강제 | 실측 모델 요구량과 무관한 장치 제한 | 제거. CUDA 이용 가능 여부와 실제 preflight 실행으로 확인 |
| P2 | 종료 직전의 완성된 `.part` 파일에 EOF Range 재요청 | 재실행 시 416 응답으로 복구가 막힐 수 있음 | checksum이 맞는 완성 partial은 HTTP 요청 없이 게시. 잘못된 checksum을 우회하지 않음 |
| P2 | 10초마다만 다운로드 진행 출력 | 빠르게 끝난 파일이 로그에서 빠져 진행 상태가 불명확 | 파일 시작·완료·검증 상태와 장면 번호/전체 개수 출력. preflight의 전체 hash 목록은 파일에 유지하고 콘솔에서는 생략 |
| P2 | 한 update만으로 잔여 시간을 추정해 자동 중단 가능 | 초기 로딩·무작위 mask에 따른 흔들림이 전체 ETA로 확대됨 | 20 update 미만 추정을 provisional로 표시하고 이를 근거로 defer하지 않음. 로딩 시간·cache hit·실제 batch 크기도 별도 기록 |

기존 포맷의 `last.pt`에는 best 복사본이 없을 수 있다. 이 경우 `model.pt`의 selected step과 resumable best step이 맞는지 확인하며, 불일치하면 임의로 checkpoint를 선택하지 않고 명시적으로 실패한다.

## 모듈과 실행 목적

| 모듈 | 기능 |
|---|---|
| `tap_checkpoint.py` | live/선택 checkpoint의 일관된 snapshot, 모델 연산 소스 호환성 검사 |
| `tap_batching.py` | 실제 query microbatch, 독립 상태 scatter/gather, per-query loss, interval loss 기록 |
| `tap_readout.py` | 전체 heatmap 평균·최대점·최대점 주변 평균 비교 |
| `tap_calibration.py` | 가림 Brier/ECE·혼동행렬, 위치 NLL·68/95% Gaussian coverage |
| `tap_diagnostics.py` | 위치/가림 분리, oracle, 시간 구간별 성능, 좌표 overlay, 읽기 쉬운 보고서 |
| `tap_probes.py` | train-only 작은 표본 암기 검사, module별 gradient, 순차/배치 처리량 비교 |
| `scripts/diagnose_tapvid.py` | 위 기능을 별도 실행하는 CLI |

진단과 probe는 train/validation 역할만 허용한다. DAVIS test 파일은 해시 계산을 포함해 진단에서 읽지 않으며, 선택된 입력 파일만 기존 manifest hash와 대조한다. 이후 benchmark의 일반 평가·무결성 검사는 여전히 전체 역할의 파일을 확인할 수 있다.

## 기존 checkpoint 진단

다음은 새 코드를 `/workspace/deep-ats-review-v2`에 풀었다고 가정한 예다. 기존 프로젝트의 경로는 서버 배치에 맞게 지정한다. GPU를 점유하지 않도록 첫 진단은 CPU로 실행한다. 진단 output은 새 이름을 사용한다.

```text
/workspace/deep-ats-code-results/.venv4090/bin/python /workspace/deep-ats-review-v2/scripts/diagnose_tapvid.py checkpoint --data-manifest /workspace/deep-ats-code-results/data/external/tapvid_4090_s128/prepared256_t60/manifest.json --checkpoint /workspace/deep-ats-code-results/outputs/tapvid_4090_s128_v1/seed17/train/last.pt --output-root /workspace/deep-ats-review-v2/outputs/diagnostic_01 --split validation --max-clips 2 --max-queries 4 --device cpu
```

주요 결과:

- `diagnostic_report.md`: readout/가림 조건별 AJ·위치 정확도·가림 정확도·위치 오차.
- `components.csv`: 전체와 query 후 1–15, 16–30, 31–60, 61프레임 이후 구간별 결과.
- `calibration.csv`: visibility Brier/ECE/혼동행렬과 위치 covariance coverage/NLL.
- `*_overlay.png`: GT, 필터 좌표, 관측 평균, heatmap 최대점의 영상 overlay.
- `diagnostic_snapshot.pt`, `*_trace.npz`: 실제로 분석한 고정 가중치와 추적 결과.
- `diagnosis.json`: 원인 확인에 참고할 기술적 차이. 원인이 증명됐다는 판정은 하지 않음.

동일 checkpoint로 `--split train`도 따로 실행하면 학습 표본과 validation의 차이를 볼 수 있다. 모든 점수는 선언된 작은 subset의 진단값이다. train/validation 장면 길이와 표본 구성이 다르므로 그 차이를 일반화 격차의 완전한 분해로 해석하지 않는다.

원시 관측/peak/local readout은 **원래 global-mean 필터 경로에서 생성한 logits**를 재사용한다. 이 값만 바꾼 end-to-end 추적기 실행 결과가 아니다. `oracle_visibility_DIAGNOSTIC_ONLY`는 예측 후 정답 가림을 사용한 진단용 상한 참고값이며 실제 모델 성능으로 보고하지 않는다. 10 percentage point 차이에 대한 자동 메시지는 원인 후보를 제안하는 고정 heuristic이며 유의성 검사나 모델 선택 규칙이 아니다.

## 다음에 확인할 순서

1. **학습 가능한가:** `overfit`으로 train query 하나를 반복 학습한다. 이조차 잘 맞추지 못하면 일반화 논의 전에 objective·readout·filter를 점검한다.
2. **위치인가 가림인가:** predicted visibility와 oracle visibility, within-threshold metric과 OA를 대조한다.
3. **시각 측정인가 필터인가:** 원시 measurement와 filtered 좌표의 오차를 비교한다.
4. **heatmap 평균이 문제인가:** peak/local 위치가 전체 평균보다 좋은지 본다. 실제 readout 교체와 재학습은 그 결과로 별도 실험한다.
5. **시간이 지나며 무너지는가:** 시간 구간별 위치 오차·가림 오류와 covariance coverage를 본다.
6. **연산 병목은 무엇인가:** resources의 로딩 시간·batch 크기와 별도 throughput probe를 확인한다.

새 checkout 디렉터리에서 다음처럼 실행한다. `<original manifest>`는 기존 서버의 절대 경로다.

```text
/workspace/deep-ats-code-results/.venv4090/bin/python scripts/diagnose_tapvid.py overfit --data-manifest <original manifest> --output-root outputs/overfit_01 --steps 200 --max-frames 16 --device cpu
/workspace/deep-ats-code-results/.venv4090/bin/python scripts/diagnose_tapvid.py benchmark --data-manifest <original manifest> --output-root outputs/benchmark_01 --batch-size 4 --max-frames 60 --feature-dim 64 --hidden-size 96 --device cuda
```

GPU throughput probe는 기존 학습과 같은 GPU에서 동시에 실행하면 서로의 측정에 영향을 준다. 실제 속도 비교는 기존 실행 종료 후 측정한다. overfit은 새 작은 모델을 train 표본 하나에 맞추는 검사이며 현재 서버 모델의 추가 학습 또는 성능 보장이 아니다.

## 새 학습의 opt-in 배치 설정

`configs/tapvid_4090_batch4.json`은 기존 데이터·모델·query 수·learning-rate schedule을 유지하고 실제 microbatch 크기를 4로 지정한다. 기본 v1 profile은 batch size 1로 유지한다. query clock이 다르면 별도 그룹이므로 실제 평균 batch 크기는 4보다 작을 수 있다. `resources.json`에 이를 기록한다.

```text
/workspace/deep-ats-code-results/.venv4090/bin/python scripts/run_tapvid_4090.py all --config configs/tapvid_4090_batch4.json --raw-root /workspace/deep-ats-code-results/data/external/tapvid_4090_s128 --prepared-root /workspace/deep-ats-code-results/data/external/tapvid_4090_s128/prepared256_t60
```

새 checkout의 상대 data 경로는 기존 데이터를 자동으로 가리키지 않는다. 위 `--raw-root`와 `--prepared-root`는 **기존 서버 데이터의 절대 경로**를 지정해 재다운로드를 피한다. output root는 새 실험 폴더로 둔다. 처음부터 새 학습이며 기존 run에 `--resume`하는 방법은 아니다. 기존 source fingerprint 검사를 끄거나 과거 결과를 재표기하지 않는다.

배치 loss와 gradient는 CPU에서 순차 reference와 수치 허용오차 안의 동등성을 검사한다. floating-point 합산 순서가 달라 두 학습 곡선이 bitwise 동일하다고 주장하지 않는다. 실제 CUDA deterministic backward와 VRAM/속도는 서버에서 검증해야 한다. 사용 연산의 지원 범위는 [PyTorch 2.12 deterministic algorithms 문서](https://docs.pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html)를 확인했다. 문서 확인만으로 GPU 실행 검증을 대체하지 않는다.

## 남아 있는 모델 한계

낮은 AJ를 위 저장·로깅 문제의 결과라고 단정하지 않는다. 작은 CNN, 사전학습 부재, 초기 query appearance만 사용하는 구조, diffuse/multimodal heatmap의 전체 평균, learned R와 visibility가 결합된 filter update는 별도 성능 실험이 필요한 한계다. 이번 구현은 원인 분리와 실행 효율·안정성을 개선하며, 새로운 모델의 성능 향상·수렴·최상위 학회 경쟁력을 입증하지 않는다.

## 이번 로컬 검증 결과

- 전체 테스트: **205 passed, 2 skipped**. 건너뛴 두 검사는 CUDA 하드웨어가 필요한 검사다.
- 기존 포맷의 로컬 3-update checkpoint에서 validation 1개 영상·2개 query 진단을 실행하고 source/artifact 검사를 통과했다. 서버 1,500-update 모델의 진단 결과가 아니다.
- 실제 Kubric train 장면의 **query 하나·8프레임**을 새 F=16/H=24 모델로 100 update 반복 학습했다. query 이후 평가 프레임 7개에서 AJ 100%, within-threshold 100%를 얻었다. 짧은 학습 표본을 외울 수 있다는 검사이며 일반화 성능이 아니다. 60/120프레임 추적의 성공을 입증하지 않는다.
- **CPU throughput은 배치가 더 느렸다.** 동일 입력·mask·초기 가중치, 256×256, 8프레임, F=16/H=24, query 4개, thread 2, warmup 1회 후 측정 3회에서 순차 update 평균 **0.311초**, microbatch 평균 **0.382초**였다. 이 한 CPU probe에서는 microbatch가 약 23% 느렸다. CUDA 속도 향상은 아직 확인하지 않았으므로 기본 profile의 batch 1을 유지하고 batch 4는 opt-in으로 제공한다.
- 위 진단·overfit·benchmark artifact의 현재 source 검사와 저장 파일 hash 검사를 모두 통과했다. 결과는 `outputs/tapvid_review_verified_20261022/`에 있다.

연속 좌표 Gaussian NLL은 확률밀도에 대한 값이므로 음수가 가능하다. overfit probe에서 total loss가 음수가 된 것 자체를 오류로 판정하지 않으며, 비유한 loss·gradient는 별도 검사한다. 작은 표본 암기 성공이나 unit-test 통과를 validation/test 성능 향상으로 바꾸어 보고하지 않는다.
