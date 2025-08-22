# 개발자 가이드: 모델 추출 공격 저장소

이 가이드는 모델 추출 공격을 구현하고 실행하기 위한 저장소의 구조, 워크플로 및 일관된 개발을 위한 지침을 제공합니다.

## 1. 저장소 구조

```
.
├── main.py                     # 실험 실행을 위한 메인 진입점
├── requirements.txt            # Python 의존성
├── configs/                    # 실험 설정 파일 (YAML)
│   ├── default.yaml            # 기본 설정
│   ├── prada_mnist_experiment.yaml # PRADA MNIST 실험 예시 설정
│   ├── tramer_mnist_experiment.yaml # Tramer MNIST 실험 예시 설정
│   └── knockoff_mnist_experiment.yaml # Knockoff MNIST 실험 예시 설정
├── ckpts/                      # 사전 훈련된 모델 체크포인트
├── data/                       # 데이터셋 (예: MNIST, CIFAR10)
├── runs/                       # 실험 결과 디렉토리 (로그, 메트릭, 훈련된 모델)
│   ├── prada_mnist_experiment/
│   ├── tramer_mnist_experiment/
│   └── ...
├── model_extraction_attack/    # 프로젝트의 메인 Python 패키지
│   ├── __init__.py
│   ├── builder.py              # 모델, 데이터 로더, 공격 인스턴스 빌드
│   ├── config.py               # 설정 로드 및 파싱 처리
│   ├── crafter.py              # 합성 샘플 (예: 적대적 예시) 생성을 위한 유틸리티 함수
│   ├── data.py                 # 데이터 로드 및 전처리 유틸리티
│   ├── metrics.py              # 다양한 평가 메트릭 계산을 위한 함수
│   ├── models.py               # 신경망 모델 정의 (타겟 및 대체 모델)
│   ├── oracles.py              # 피해자 모델 쿼리를 위한 블랙박스 오라클 인터페이스 정의
│   ├── runner.py               # 실험 실행 오케스트레이션
│   ├── utils.py                # 일반 유틸리티 함수 (예: 로깅, 시딩, soft_cross_entropy)
│   └── attacks/                # 다양한 모델 추출 공격 구현
│       ├── __init__.py         # 공격 클래스 등록
│       ├── base_attack.py      # 모든 공격의 기본 클래스
│       ├── prada.py            # PRADA 공격 구현
│       ├── tramer.py           # Tramer 공격 구현
│       ├── knockoff.py         # Knockoff 공격 구현
│       ├── ...                 # 다른 공격 구현
│       └── registry.py         # 공격 클래스를 위한 중앙 레지스트리
└── tests/                      # 단위 및 통합 테스트
```

## 2. 공격 구현 및 실행 워크플로

### 2.1 새로운 공격 추가

새로운 모델 추출 공격을 추가하려면:

1.  **새로운 공격 파일 생성:** `model_extraction_attack/attacks/` 디렉토리에 새로운 Python 파일 (예: `my_new_attack.py`)을 생성합니다.
2.  **공격 클래스 구현:**
    *   공격 클래스는 `model_extraction_attack.attacks.base_attack.BaseAttack`을 상속해야 합니다.
    *   필요한 매개변수 (학생 모델, 오라클, 쿼리 로더, 장치, 그리고 공격별 설정)를 받도록 `__init__` 메서드를 구현합니다.
    *   공격의 핵심 로직을 포함하는 `run()` 메서드를 구현합니다. 이 메서드는 훈련된 학생 모델과 총 쿼리 수를 반환해야 합니다.
    *   공격이 라운드별 메트릭을 `self.metrics_history` (딕셔너리 목록)에 기록하고, `run()` 메서드 끝에서 `self.run_dir` 내의 JSON 파일에 저장하는지 확인하십시오. 이는 일관된 시각화를 위해 중요합니다.
3.  **공격 등록:**
    *   공격 클래스 정의 위에 `@register_attack("my_new_attack_name")` 데코레이터를 추가합니다.
    *   `model_extraction_attack/attacks/__init__.py`에서 새로운 공격 모듈을 임포트합니다 (예: `from . import my_new_attack`). 이는 공격이 `builder`에 의해 등록되고 검색 가능하도록 보장합니다.
4.  **설정 파일 생성:**
    *   `configs/` 디렉토리에 새로운 YAML 파일 (예: `my_new_attack_experiment.yaml`)을 생성합니다.
    *   `attack` 섹션 아래에 `name: my_new_attack_name`을 포함하여 공격에 필요한 모든 매개변수를 정의합니다.
    *   피해자, 학생, 쿼리 데이터 설정을 지정합니다.
5.  **`builder.py` 업데이트 (필요한 경우):**
    *   공격이 `build_attack`의 일반적인 `kwargs`로 처리되지 않는 특별한 처리나 추가 매개변수를 필요로 하는 경우, `model_extraction_attack/builder.py`에 `if attack_name == "my_new_attack_name":` 블록을 추가해야 할 수 있습니다. 그러나 여기에 변경을 최소화하기 위해 공격의 `__init__`이 `**kwargs`를 통해 매개변수를 받을 수 있도록 유연하게 만드십시오.
6.  **테스트 추가:** `tests/` 디렉토리에 새로운 공격에 대한 단위 및 통합 테스트를 작성합니다.

### 2.2 실험 실행

특정 설정으로 실험을 실행하려면:

```bash
python main.py --config configs/your_experiment_config.yaml
```

결과 (로그, 메트릭 JSON, 훈련된 학생 모델 체크포인트)는 설정 파일에 지정된 `runs/your_experiment_name/` 디렉토리에 저장됩니다.

### 2.3 결과 시각화

여러 실험의 결과를 시각화하고 비교하려면:

1.  **메트릭 JSON 파일이 생성되었는지 확인:** 먼저 원하는 모든 실험을 실행합니다.
2.  **`prada_viz.py` 업데이트:** `prada_viz.py`의 `experiments_to_compare` 목록을 수정하여 비교하려는 각 실험의 `metrics_file` 경로와 `label`을 포함합니다.
3.  **시각화 스크립트 실행:**

    ```bash
    python prada_viz.py
    ```

    비교 플롯은 `runs/comparison_plots/`에 저장됩니다.

## 3. 일관성 가이드라인

다양한 공격 구현 전반에 걸쳐 일관성과 유지보수성을 보장하려면:

*   **BaseAttack 상속:** 모든 공격 클래스는 `model_extraction_attack.attacks.base_attack.BaseAttack`을 상속해야 합니다.
*   **`run()` 메서드 시그니처:** 공격 클래스의 `run()` 메서드는 항상 `Tuple[nn.Module, int]` (훈련된 학생 모델, 총 쿼리 수)를 반환해야 합니다.
*   **설정 기반:** 모든 공격별 매개변수는 YAML 설정 파일에 정의되어야 하며, `__init__` 메서드에 전달되는 `prada_config` (또는 PRADA가 아닌 공격의 경우 `attack_config`) 딕셔너리를 통해 접근해야 합니다.
*   **메트릭 로깅:** 라운드별 메트릭을 `self.metrics_history` (딕셔너리 목록)에 기록하고, `run()` 메서드 끝에서 이 목록을 `self.run_dir` 내의 JSON 파일 (예: `your_attack_metrics.json`)에 저장하십시오. `metrics_history`의 각 딕셔너리에는 최소한 `round`, `n_queries`, `current_labeled_dataset_size`가 포함되어야 합니다. 필요에 따라 다른 관련 메트릭을 포함하십시오.
*   **옵티마이저 및 손실 함수:** 설정의 `optimizer_type`, `optimizer_params`, `lr`, `epochs`, `label_only` 매개변수를 사용하여 학생 모델의 훈련 루프를 일관되게 설정하십시오. 소프트 레이블의 경우 `utils.py`의 `soft_cross_entropy`를 활용하십시오.
*   **데이터 처리:** 배칭을 위해 `DataLoader`를 활용하십시오. 초기 시드 샘플의 경우 `data.py`의 `make_seed_from_labeled`를 사용하십시오.
*   **모델 빌딩:** `models.py`의 `build_model`을 사용하여 학생 및 피해자 모델을 인스턴스화하십시오.

## 4. 재사용 가능한 유틸리티

`model_extraction_attack/` 패키지는 활용해야 할 여러 유틸리티 모듈을 제공합니다:

*   **`model_extraction_attack/metrics.py`:**
    *   `calculate_papernot_transferability`: Papernot 스타일 전이성 메트릭용.
    *   `test_agreement`: 테스트 세트에서 매크로 평균 F1-점수를 계산합니다.
    *   `ru_agreement`: 무작위 균일 샘플에 대한 정확도를 계산합니다.
    *   `agreement`: 두 모델 간의 일반적인 일치도.
    *   **사용법:** 공격의 `run()` 메서드 또는 헬퍼 함수에서 이러한 함수를 직접 임포트하고 호출합니다.
*   **`model_extraction_attack/models.py`:**
    *   `build_model`: 다양한 신경망 아키텍처 (예: `prada_mnist_target`, `prada_sub_cnn2`)를 생성하는 팩토리 함수.
    *   `load_checkpoint`: 사전 훈련된 모델 가중치를 로드하기 위한 함수.
    *   **사용법:** `builder.py`에서 `build_model`을 호출하여 모델을 인스턴스화합니다.
*   **`model_extraction_attack/crafter.py`:**
    *   합성 샘플 (예: `fgsm_family_crafter`, `color_aug_batch`, `jsma_batch`) 생성을 위한 함수를 포함합니다.
    *   **사용법:** 공격의 샘플 생성 로직 (예: `_create_new_samples`)에서 이러한 함수를 임포트하고 호출합니다.
*   **`model_extraction_attack/data.py`:**
    *   `get_mnist_loaders`, `get_cifar10_loaders`, `get_imagefolder_loaders`: 표준 데이터셋 로드를 위한 함수.
    *   `make_seed_from_labeled`: 레이블된 데이터에서 초기 시드 데이터셋을 생성하는 데 중요합니다.
    *   **사용법:** 데이터 로더는 일반적으로 `builder.py`에서 빌드되어 공격에 전달됩니다. `make_seed_from_labeled`는 초기 데이터 설정을 위해 공격의 `run()` 메서드 내에서 사용될 수 있습니다.
*   **`model_extraction_attack/utils.py`:**
    *   `set_seed`: 재현성을 위한 함수.
    *   `setup_logger`: 일관된 로깅을 위한 함수.
    *   `save_json`: JSON 파일에 결과를 저장하기 위한 함수.
    *   `soft_cross_entropy`: 소프트 교차 엔트로피 손실 계산을 위한 중앙 집중식 함수.
    *   **사용법:** 필요에 따라 이러한 일반 유틸리티 함수를 임포트하고 사용합니다.

---
