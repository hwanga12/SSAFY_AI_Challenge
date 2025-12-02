# 🧠 이미지를 '이해'하고 '답변'하다: SSAFY AI 챌린지 - 이미지 기반 VQA 모델 개발

| 구분 | 내용 |
| :--- | :--- |
| **프로젝트 유형** | 멀티모달 AI / 분류 (VQA) / Kaggle 기반 |
| **팀 구성** | 4인 (팀명: 상미나이) |
| **기간** | 2025.10.23 ~ 2025.10.27 (4일간) |
| **주요 역할** | 모델 선정, 파인튜닝, 최적화, 실험 설계 및 검증 |

---

# 🛠️ 1. 기술 스택 (Tech Stack)
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

# 📝 2. 프로젝트 개요 (Project Overview)

본 프로젝트는 **SSAFY 14기 AI 챌린지**의 일환으로, 이미지와 자연어 질문을 동시에 이해하여 보기(a, b, c, d) 중 정답을 선택하는 **VQA (Visual Question Answering) 모델**을 개발하는 것을 목표로 합니다.

VQA 모델은 이미지 속 상황과 객체를 '이해'하고 논리적으로 '답변'하는 **멀티모달 AI의 핵심 기술**입니다. 우리는 이 챌린지를 통해 최신 멀티모달 모델의 작동 원리를 이해하고 실제 서비스에 적용 가능한 AI 모델을 구축하는 경험을 했습니다.

* **미션:** 주어진 이미지와 자연어 질문을 입력받아 정답 선지 (a, b, c, d)를 예측하는 4지선다형 분류 문제 해결.
* **활용 기술:** 비전과 자연어의 융합을 다루는 멀티모달 파인튜닝, A100 GPU 환경 최적화, Prompt Engineering.

---

# 🗂️ 3. 폴더 구조 (Folder Structure)
📦 VQA-Challenge-TeamName/ ├── 📁 data/            # 학습 데이터 및 이미지 파일 │   ├── train.csv         # 학습용 질문/정답 쌍 데이터 │   └── images/           # 원본 이미지 파일 ├── 251023_Baseline.ipynb # 초기 베이스라인 코드 (Colab 기반) ├── training_script.py    # 최종 모델 학습 및 추론 코드 ├── submission.csv        # Kaggle 최종 제출 파일 └── README.md             # 프로젝트 설명 (현재 문서)


---

# ⚙️ 4. 모델 및 데이터 설정 (Model & Data Configuration)

## 4.1 모델 및 데이터
* **핵심 모델:** `Qwen/Qwen2.5-VL-7B-Instruct`
* **데이터셋:** 데이터 수집 미션으로 가공된 이미지-질문-정답 쌍 데이터 (퀴즈 형태)

## 4.2 평가 기준
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

# 🚀 5. 주요 변경 사항 및 최적화 (코드 업그레이드)

제공된 베이스라인 코드를 **A100 고성능 GPU** 환경에 맞춰 최대 성능을 이끌어내기 위해 아래와 같이 파인튜닝 설정을 최적화했습니다.

<table border="1" cellpadding="5" cellspacing="0">
  <thead>
    <tr>
      <th>변경 항목</th>
      <th>기존 설정 (베이스라인)</th>
      <th>개선 설정 (업그레이드)</th>
      <th>문제 해결 및 최적화 근거</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>양자화 설정</td>
      <td>4bit 양자화 (BitsAndBytesConfig)</td>
      <td>**삭제 (bf16 Full Precision)**</td>
      <td rowspan="7">아래 5.1 최적화 근거 참고</td>
    </tr>
    <tr>
      <td>모델 채택</td>
      <td>(초기 모델)</td>
      <td>`Qwen/Qwen2.5-VL-7B-Instruct`</td>
    </tr>
    <tr>
      <td>데이터 활용 범위</td>
      <td>샘플 데이터 200개 제한</td>
      <td>**전체 데이터셋 사용**</td>
    </tr>
    <tr>
      <td>학습 Epoch</td>
      <td>1 Epoch</td>
      <td>**5 Epoch**</td>
    </tr>
    <tr>
      <td>학습률 (LR)</td>
      <td>(기존 값)</td>
      <td>**2e-5** (더 안정적인 수렴 유도)</td>
    </tr>
    <tr>
      <td>이미지 해상도</td>
      <td>(기존 값)</td>
      <td>**448 x 448** (시각 정보 이해도 향상)</td>
    </tr>
    <tr>
      <td>메모리 관리</td>
      <td>없음</td>
      <td>`gc.collect()`, `torch.cuda.empty_cache()` 추가</td>
    </tr>
  </tbody>
</table>

## 5.1 최적화 근거 (Optimization Rationale)
[4bit 양자화 삭제 및 bf16 전환 이유] 4bit 양자화는 VRAM이 적은 환경(T4, 16GB 이하)에서 메모리 절약을 위해 사용되지만, 정확도가 약간 떨어질 수 있다는 단점이 있습니다. A100 GPU는 VRAM 40GB 이상을 제공하므로, 7B 모델을 bf16 (Brain Float 16) 풀 정밀도로 학습해도 충분합니다. bf16은 float16보다 정밀도를 더 유지하면서 VRAM 효율도 좋기 때문에, 모델의 최대 성능과 표현력을 확보하기 위해 양자화 설정을 삭제하고 Full Precision으로 전환했습니다.


---

# 🏆 6. 주요 성과 및 결론 (Key Achievements)
* **고성능 환경 최적화 경험:** A100 GPU 환경의 특성(넉넉한 VRAM)을 분석하여 4bit 양자화 제거, bf16 Full Precision 학습 등 **최적의 모델 학습 설정**을 성공적으로 설계하고 구현했습니다.
* **멀티모달 AI 구현 심화:** 이미지와 텍스트를 통합적으로 처리하는 **VQA 모델의 파인튜닝 파이프라인**을 성공적으로 구축하고, 최신 모델(Qwen)의 활용 능력을 입증했습니다.
* **AI 프로젝트 전 과정 경험:** 데이터 전처리, 모델 학습 및 최적화, 추론, 최종 제출까지 **AI 프로젝트의 전 과정을 경험**하며 실질적인 문제 해결 역량을 강화했습니다.