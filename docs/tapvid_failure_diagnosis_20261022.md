# AJ 20%대 결과에 대한 구현·모델 진단

2026-10-22. 서버가 삭제되었다는 사용자 확인을 반영했다. 당시 17,500 step 이후 checkpoint는 로컬에 없으므로, 그 모델의 오류를 수치로 분해했다고 주장하지 않는다. 아래에서 코드상 확정 사실, 로컬 통제 실험, 실제 20% 모델에 대한 가설을 구분한다.

## 판단

가장 먼저 의심할 부분은 **기본 시각 대응점 추정과 좌표 readout**이다. 단순히 학습이 덜 됐거나 GPU가 놀아서 AJ가 낮았다고 설명할 근거는 부족하다. 여러 peak를 평균내는 readout, 운동 예측을 사용하지 않는 시각 매칭, 고정된 최초 query descriptor와 작은 CNN이 함께 사용된다. 이들은 현재 코드에서 확인되지만, 각 요소가 당시 AJ를 몇 점 낮췄는지는 복원할 수 없다.

좌표 축·가림 반전·공식 지표 계산이 잘못되어 무조건 20%에 막힌다는 증거는 발견하지 못했다. 실제 로컬 데이터에서 이런 후보를 검사했다. 이것은 전체 구현의 무오류 증명이 아니다.

## 1. 현재 로그는 ATS의 관측 선택 성능을 평가하지 않는다

`configs/tapvid_4090.json`은 validation 관측 비율을 1.0으로 설정하고 `tap_training.py:validation_scores`는 기본 uniform 평가를 사용한다. 모든 프레임을 읽으므로 어느 시점을 건너뛸지 선택하는 자유도가 없다. 따라서 이 로그는 우선 **전체 관측 기본 추적기**의 품질을 보여준다. 낮은 AJ를 ATS 관측 정책 자체의 실패라고 단정할 수 없다.

사용자가 제공한 최고 기록은 17,500 step의 0.21014, 즉 AJ 21.014%였다. 이는 Kubric custom validation의 일부 영상·query 결과이며 DAVIS test 결과가 아니다.

## 2. 가장 구체적인 설계 문제: 전체 heatmap 좌표 평균

`src/ats_vision/tap_model.py:170–171`은 전체 화면의 softmax 확률에 좌표를 곱해 평균을 취한다. 정답이 있는 peak가 가장 높더라도, 나머지 수천 개 위치에 남은 확률이나 다른 peak 때문에 결과 좌표가 정답에서 멀어질 수 있다.

확인한 통제 결과:

| 통제 | 전체 평균 | peak | peak 주변 평균 |
|---|---:|---:|---:|
| 64×64 map의 동일한 두 peak; 왼쪽을 지정 정답으로 둔 구성 예제 | 64 px 오차 | 0 px | 0 px |
| 로컬 3-update checkpoint, 동일 validation 이미지에서 query를 다시 찾기; 4 query의 오차 중앙값 | 100.35 px | 1.22 px | 2.12 px |

두 번째 검사는 움직임·가림 변화 없이 **동일 이미지**를 다시 입력했다. 대응 peak는 query 근처인데 전체 평균 좌표는 멀어지는 실행 사례를 확인했다. 사용한 모델은 예전 로컬 F16/H24·3-update 실행이며, 삭제된 F64/H96·17,500-update 모델이 아니다. 초기 또는 약한 heatmap에서 생기는 실패 방식의 실제 재현이지, 당시 20% 모델의 원인 비중 추정이 아니다.

단순 argmax로 바꾸면 subpixel 정보가 사라지고 잘못된 peak 선택도 발생할 수 있다. frozen logits의 peak/local 성적이 높다는 사실만으로 재학습한 새 모델의 성능 향상을 확정할 수 없다.

## 3. 운동 상태가 대응점 검색을 도와주지 않는다

`tap_model.py:153–160`의 시각 logits 입력은 현재 프레임 특징, 최초 descriptor, cosine similarity뿐이다. 예측 위치·속도·GRU context는 해당 logits 계산에 들어가지 않는다. 그 뒤 계산하는 R/visibility와 최종 filter에는 운동 상태가 관여한다.

서버와 같은 F64/H96 모델의 연산 구조를 별도로 검사했다. 이미지와 query descriptor를 고정하고 belief의 위치·속도와 context를 바꿔도 logits와 raw measurement가 **정확히 같았다**. 즉, 비슷한 물체가 여러 위치에 있을 때 운동 예측으로 대응 후보를 좁히는 기능이 현재 시각 매칭부에는 없다. 이는 코드상 확정된 성질이며 실제 오류 빈도는 알 수 없다.

또한 `initialize`, `predict`, `observe`를 따라 확인하면 최초 query descriptor가 전체 추적 동안 바뀌지 않는다. 관측 후 descriptor가 그대로임도 검사했다. 외형·크기·회전 변화에 대한 적응은 고정된 descriptor와 학습된 feature의 일반화 능력에 의존한다.

## 4. 기본 시각 모델과 학습 설정의 한계

- 서버 모델은 **188,202 parameter**, 3-convolution encoder다. Convolution만 따라 계산한 기하학적 receptive field는 encoder 17×17 px, decoder까지 25×25 px다. 다만 GroupNorm은 이미지 전체의 group 통계에 의존하므로 이 수치를 엄밀한 입력 의존성의 상한으로 해석하면 안 된다. 명시적인 multi-scale·운동 prior 기반 대응점 탐색이 없는 점과 구분한다.
- 사전학습 backbone, multi-scale 대응점 탐색, temporal visual memory가 없다. GRU는 좌표·innovation·noise·visibility 통계로 갱신되며 시각 feature map을 기억하는 구조가 아니다.
- 이미지 crop/scale/color augmentation 없이 고정된 window에서 query와 관측 mask를 무작위로 뽑는다. 서버 설정은 총 128 source scene, 그중 validation 13 scene이며 로그 계산에는 최대 8 scene·영상당 16 query를 사용한다.
- 한 update의 8 query는 같은 영상에서 나온다. 8개의 서로 다른 영상으로 구성된 batch와 같은 데이터 다양성을 제공하지 않는다.

이 사실들 때문에 데이터·표현·대응점 추정의 병목을 의심할 이유는 있다. 그러나 parameter가 적다는 이유만으로 당시 21%를 설명하거나 큰 backbone이 반드시 해결한다고 주장하지 않는다.

## 5. 학습·평가 horizon과 관측 mask 차이

서버 profile은 학습 window 60프레임, validation은 전체 120프레임을 유지한다. 학습에서 가능한 query 이후 최대 길이는 59프레임, 평가에서는 119프레임이다. 후반 drift·외형 변화·재등장 난도가 달라질 수 있다.

학습 query의 관측 비율은 25/50/75/100%에서 뽑고 관측 시점을 무작위로 고른다. 모니터링은 100% 관측이다. 이는 의도한 여러 budget 학습이지만 전체 관측 정확도만 학습하는 조건과 다르다. 이 차이의 영향은 통제된 비교 없이는 확정할 수 없다.

로그는 기본 visibility threshold 0.5로 계산한다. 학습 완료 뒤의 선택 단계는 별도의 validation threshold grid를 사용한다. 따라서 로그 AJ와 최종 threshold를 적용한 성적이 달라질 수 있으나, 삭제된 모델에서 그 차이를 계산할 수 없다.

17,500 step의 cosine LR은 약 **4.137e-5**로 최대 3e-4의 약 13.8%다. 끝부분의 개선이 작아지는 데 영향을 줄 수 있지만, LR 일정이 낮은 절대 성능의 주원인이라고 단정할 수는 없다.

## 6. Q/R·filter·visibility는 검사 대상이지만 아직 주범으로 확인되지 않았다

현재 R은 heatmap spread에서 만들어지고, visibility가 낮으면 effective R이 커진다. 넓거나 잘못된 heatmap이 낮은 신뢰도로 해석되어 시각 관측의 영향이 작아질 수 있다. noisy observation을 낮게 신뢰하는 동작 자체는 filter 오류가 아니다.

로컬 3-update 통제에서는 오히려 global raw measurement보다 filtered localization이 나았다. 따라서 filter를 제거하면 해결된다고 주장할 증거는 없다. 그 통제의 validation에서 effective measurement 표준편차 중앙값은 약 128 px였지만, 이 수치를 삭제된 서버 모델에 적용하면 안 된다.

## 7. 손실·gradient 확인

좌표 손실 + visibility BCE + 0.1×measurement NLL + heatmap CE를 함께 최적화하고 gradient norm을 5로 제한한다.

예전 로컬 3-update 모델의 한 학습 query·16프레임 full-read prefix에서 objective 가중치를 반영한 gradient norm은 다음과 같았다.

| 항 | gradient norm |
|---|---:|
| heatmap CE | 23.39 |
| filtered coordinate | 9.78 |
| visibility | 0.79 |
| 0.1×measurement | 0.65 |

해당 단일 prefix의 합 gradient norm은 30.91이라 norm 5 제한을 받는 크기였다. 실제 학습은 여러 query를 평균하므로 이 값으로 clipping 빈도나 당시 학습의 비율을 추정하지 않는다. CE와 coordinate gradient cosine은 약 +0.598이었다. 이 검사에서는 두 항이 서로 반대 방향으로 학습한다는 주장이 지지되지 않았다. 여러 loss가 있다는 이유만으로 충돌을 주범으로 삼을 수 없다.

Full-read에서는 forecast visibility head가 호출되지 않아 그 head의 gradient가 0인 것이 정상이다. 50% 관측 통제에서는 그 head에 gradient가 전달되는 것도 검사했다.

## 8. 반증·구현 검증

- 로컬 raw Kubric scene 0000/0001에서 원본 좌표를 독립적으로 변환한 결과와 저장 NPZ가 정확히 같았다. 각각 전체 validation 1 clip, training window 5개를 확인했다. 첫 RGB frame resize도 정확히 일치했다.
- 원 저자 parser의 원문은 `visibility = np.transpose(np.logical_not(visibility), (1, 0))`이다. released field를 occlusion으로 읽는 현재 규칙과 맞는다. [공식 parser](https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/kubric_movif_dataset.py)
- 실제 로컬 checkpoint의 train/validation 예측을 pinned 공식 TAP-Vid evaluator와 비교한 최대 metric 차이는 **0.0**이었다.
- 이전의 실제 train 1 query·8프레임 과적합 weight를 현재 코드로 재실행해 **AJ 100%**를 재확인했다. 평가한 future frame은 7개뿐이며 일반화 증거가 아니다. 코드나 metric에 고정적인 20% 상한이 있는 것은 아니다.
- 64×64 정답 cell centre와 정답 visibility를 넣는 label-only control의 로컬 validation AJ는 75.24%였다. 이는 grid의 단순 수치적 제한이 20%라는 설명을 반박하는 통제다. continuous soft-argmax의 실제 상한으로 해석하지 않는다.
- LK 진단 control은 같은 로컬 train clip에서 AJ 57.43%, validation clip에서는 17.40%였다. LK는 이 데이터로 학습하지 않으므로 두 scene의 움직임·가림·horizon 차이가 매우 클 수 있음을 보여준다. train–validation gap만 보고 과적합을 확정하면 안 된다. 이 수치도 사용자의 8-video 서버 validation 성적과 직접 비교하지 않는다.
- 전체 테스트 **241 passed, CUDA 전용 3 skipped**. 이는 테스트한 구현 계약의 확인이며 모델 품질이나 모든 오류의 부재를 보증하지 않는다.

## 9. 수정 우선순위

1. 고정된 데이터·초기 weight·학습량으로 global expectation과 mode 주변 subpixel readout을 비교한다. 단순 argmax 교체의 점수를 모델 개선으로 포장하지 않는다.
2. 시각 대응점 추정에 운동 prior/local search와 외형 적응을 넣는 ablation을 수행한다. 이는 설계 변경 후보이며 개선이 보장되지 않는다.
3. 전체 관측 기본 추적기를 먼저 검증하고, 그 후 동일 관측 예산의 ATS와 uniform을 비교한다. 기본 시각 대응점이 부정확하면 정교한 관측 정책만으로 해결하기 어렵다.
4. train–validation을 같은 horizon으로 비교하고, full-read 학습 vs mixed-mask 학습을 분리한다. scene 난도와 데이터 다양성도 함께 기록한다.
5. raw/filtered/oracle-visibility 비교에서 실제 손해가 확인될 때 Q/R·visibility calibration을 수정한다. 근거 없이 filter를 제거하거나 서버 학습을 오래 반복하지 않는다.

직전 속도 개선은 feature 계산 공유와 실행 방식의 변경이다. 손실·gradient parity를 유지하도록 만들었으므로 이 정확도 문제를 해결하는 설계 수정으로 볼 수 없다.

## 추가한 재사용 진단 모듈

`src/ats_vision/tap_failure_audit.py`와 `scripts/diagnose_tapvid.py failure-audit`를 추가했다. 고정 weight에서 train/validation의 readout·visibility·horizon·속도·재등장·self-match·R/gain·gradient를 검사하고, final-test 파일을 읽거나 weight를 바꾸지 않는 테스트를 통과했다. 현재 서버를 다시 요청하기 위한 절차가 아니라, 이후 새 모델을 학습할 때 같은 진단을 재사용하기 위한 구현이다.

증거 파일은 `outputs/tapvid_failure_diagnosis_20261022/`에 있다. 특히 `structural_controls.json`, `source_checks/raw_prepared_parity.json`, `overfit_replay.json`, `local_3step_audit/`를 참고한다. 로컬 3-update 결과는 당시 20% 서버 모델의 결과가 아니다.
