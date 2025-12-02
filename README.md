# 이미지를 '이해'하고 '답변'하다: SSAFY AI 챌린지 - 이미지 기반 VQA 모델 개발

![demo](assets/image%20(3).png)

| 구분 | 내용 |
| :--- | :--- |
| **프로젝트 유형** | 멀티모달 AI / 분류 (VQA) / Kaggle 기반 |
| **팀 구성** | 4인 (팀명: 상미나이) |
| **기간** | 2025.10.23 ~ 2025.10.27 (4일간) |
| **주요 역할** | 모델 선정, 파인튜닝, 최적화, 실험 설계 및 검증 |

---

# 1. 기술 스택 (Tech Stack)
<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <th>분류</th>
    <th>주요 환경 및 컴포넌트</th>
  </tr>
  <tr>
    <td>
      - 개발 언어 : Python<br>
      - 프레임워크 : PyTorch<br>
      - 라이브러리 : HuggingFace (Transformers, Datasets), PEFT (LoRA)
    </td>
    <td>
      - 모델 : **Qwen2.5-VL-7B-Instruct**<br>
      - GPU 환경 : **SSAFY SSH GPU A100 (VRAM 40GB+)**<br>
      - 개발 도구 : Google Colab (초기), Jupyter Notebook
    </td>
  </tr>
</table>

---

# 2. 프로젝트 개요 (Project Overview)

본 프로젝트는 **SSAFY 14기 AI 챌린지**의 일환으로, 이미지와 자연어 질문을 동시에 이해하여 보기(a, b, c, d) 중 정답을 선택하는 **VQA (Visual Question Answering) 모델** 개발을 목표.

VQA 모델은 이미지 속 상황과 객체를 '이해'하고 논리적으로 '답변'하는 **멀티모달 AI의 핵심 기술**이다. 본 챌린지를 통해 최신 멀티모달 모델의 작동 원리를 이해하고 실제 서비스에 적용 가능한 AI 모델을 구축하는 경험을 완료.

* **미션:** 주어진 이미지와 자연어 질문을 입력받아 정답 선지 (a, b, c, d)를 예측하는 4지선다형 분류 문제 해결.
* **활용 기술:** 비전과 자연어의 융합을 다루는 멀티모달 파인튜닝, A100 GPU 환경 최적화, Prompt Engineering.

---

# 3. 모델 및 데이터 설정 (Model & Data Configuration)

## 3.1 모델 및 데이터
* **핵심 모델:** `Qwen/Qwen2.5-VL-7B-Instruct`
* **데이터셋:** 데이터 수집 미션으로 가공된 이미지-질문-정답 쌍 데이터 (퀴즈 형태)

## 3.2 평가 기준
<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>항목</th>
      <th>내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>정확도 (Accuracy)</td>
      <td>모델의 정답 예측률 (Kaggle 리더보드 점수)</td>
    </tr>
    <tr>
      <td>모델의 효율성</td>
      <td>학습/추론 속도 및 자원 활용 최적화</td>
    </tr>
    <tr>
      <td>재현성</td>
      <td>제출 코드 기반 결과 재현 가능성</td>
    </tr>
  </tbody>
</table>

---

# 4. 주요 변경 사항 및 최적화 (코드 업그레이드)

제공된 베이스라인 코드는 VRAM이 제한적인 환경(Colab T4 등)에 맞춰 4bit 양자화와 낮은 설정(1 Epoch, 샘플 데이터 200개)으로 구성되어 있었다. 하지만 우리는 **A100 GPU**라는 최상위 환경을 확보했으므로, 모델의 성능을 극대화하고 안정적인 수렴을 달성하기 위한 대대적인 코드 업그레이드를 진행.

## 최적화 요약 (Optimization Summary)
<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>변경 항목</th>
      <th>기존 설정 (베이스라인)</th>
      <th>개선 설정 (업그레이드)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>양자화 설정</td>
      <td>4bit 양자화 (BitsAndBytesConfig)</td>
      <td><b>삭제 (bf16 Full Precision)</b></td>
    </tr>
    <tr>
      <td>모델 채택</td>
      <td>(초기 모델)</td>
      <td>`Qwen/Qwen2.5-VL-7B-Instruct`</td>
    </tr>
    <tr>
      <td>데이터 활용 범위</td>
      <td>샘플 데이터 200개 제한</td>
      <td><b>전체 데이터셋 사용</b></td>
    </tr>
    <tr>
      <td>학습 Epoch</td>
      <td>1 Epoch</td>
      <td><b>5 Epoch</b></td>
    </tr>
    <tr>
      <td>학습률 (LR)</td>
      <td>(기존 값)</td>
      <td><b>2e-5</b></td>
    </tr>
    <tr>
      <td>이미지 해상도</td>
      <td>(기존 값)</td>
      <td><b>448 x 448</b></td>
    </tr>
    <tr>
      <td>메모리 관리</td>
      <td>없음</td>
      <td>`gc.collect()`, `torch.cuda.empty_cache()` 추가</td>
    </tr>
  </tbody>
</table>

## 4.1 문제 상황 및 해결 과정 (Thought Process)

### 1. **문제 상황: VRAM 스펙과 정확도 Trade-off 불일치**
* 베이스라인은 메모리 절약을 위해 **4bit 양자화**를 사용했으나, 이는 모델의 **표현력과 정확도 저하**를 유발할 수 있는 설정.
* **A100 GPU (VRAM 40GB+)** 환경을 확보했으므로, 굳이 정확도를 희생할 이유가 사라짐.

### 2. **해결 전략: Full Precision으로 최대 성능 확보**
* **bf16 Full Precision 전환:** 4bit 양자화 설정을 제거하고, **bf16 (Brain Float 16) 정밀도**로 전환. 7B 모델을 VRAM 제약 없이 학습시켜 모델의 **최대 정밀도**를 이끌어냄.
* **학습 범위 확장:** 안정적인 수렴을 위해 Epoch을 1회에서 **5회**로 대폭 증가시키고, **데이터셋 샘플 제한(200개)**을 해제하여 전체 데이터를 활용, 모델의 일반화 성능을 높임.

### 3. **안정성 및 품질 확보를 위한 추가 조치**
* **고해상도 적용:** VQA 모델의 핵심인 시각 정보 이해도를 높이기 위해 이미지 해상도를 **448x448**로 상향.
* **안정적인 수렴:** 학습 범위 확장(5 Epoch)에 맞추어 Learning Rate를 **2e-5**로 낮춰, 급격한 손실 변화를 방지하고 더 안정적인 최적점으로 수렴하도록 유도.
* **환경 안정성:** 학습 중 VRAM 누수 문제를 방지하기 위해 각 Epoch 시작 전에 명시적으로 **메모리 정리 코드**(`gc.collect()`, `torch.cuda.empty_cache()`)를 삽입하여 환경의 안정성을 유지.

---

# 5. 주요 성과 및 결론 (Key Achievements)

* **성능 향상:** 팀의 정확도를 기존 0.75 수준에서 **0.94**까지 끌어올려 **약 25%의 성능 향상** 달성.
* **최종 순위:** 팀 최종 순위 **Top 50** 달성.
* **최적화 능력 검증:** A100 GPU 환경의 이점을 극대화하기 위해 양자화 설정, 정밀도, 데이터셋 활용 범위 등 하이퍼파라미터를 최적화하는 **AI 엔지니어링 역량**을 검증.
* **멀티모달 AI 구현 심화:** 이미지와 텍스트를 통합 처리하는 VQA 모델 파이프라인을 성공적으로 구축.

| 과정 | 중간 정확도 | 최종 정확도 |
| :---: | :---: | :---: |
| 중간 과정 | 0.75 | 0.94 |

![demo](assets/image%20(1).png) (중간 과정)
![demo](assets/image%20(2).png) (최종 정확도)
