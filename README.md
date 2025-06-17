## 2025-NLP Project

### 2022113597 김민성

### 2022113594 홍승민

# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

## 1. 실행 방법 (Google Colab 기준)

1. `!git clone https://github.com/chrismskim/2022113597KimMinseong.git`
2. `%cd 2022113597KimMinseong`
3. `!pip install -r requirements.txt`

---

## 2. 파일 구조

```
2022113597KimMinseong/
├── classifier.py
├── config.py
├── datasets.py
├── env.yml
├── evaluate_paraphrase.py
├── evaluate_sonnet.py
├── evaluation.py
├── LICENSE
├── optimizer.py
├── optimizer_test.npy
├── optimizer_test.py
├── paraphrase_detection.py
├── prepare_submit.py
├── README.md
├── requirements.txt
├── sanity_check.py
├── sonnet_generation_basic.py
├── sonnet_generation_Lora_Peft.py
├── utils.py
├── data/
│   ├── ids-cfimdb-dev.csv
│   ├── ids-cfimdb-test-student.csv
│   ├── ids-cfimdb-train.csv
│   ├── ids-sst-dev.csv
│   ├── ids-sst-test-student.csv
│   ├── ids-sst-train.csv
│   ├── imdb_test.csv
│   ├── imdb_train.csv
│   ├── quora-dev.csv
│   ├── quora-test-student.csv
│   ├── quora-train.csv
│   ├── sonnets.txt
│   ├── sonnets_held_out.txt
│   ├── sonnets_held_out_dev.txt
│   └── TRUE_sonnets_held_out_dev.txt
├── models/
│   ├── base_gpt.py
│   └── gpt2.py
├── modules/
│   ├── attention.py
│   └── gpt2_layer.py
├── predictions/
│   └── README
```

---

## 3. Part 1 실행 방법 (테스트 결과 예시 포함)

1. optimizer.py 구현을 테스트

   - 실행 방법: `!python optimizer_test.py`
   - 출력 예시:
     ```
     tensor([...])  # (ref 값)
     tensor([...])  # (actual 값)
     Optimizer test passed!
     ```

2. GPT 모델 구현을 테스트

   - 실행 방법: `!python sanity_check.py`
   - 출력 예시:
     ```
     Your GPT2 implementation is correct!
     ```

3. 모델을 사용한 감정 분류 수행
   - 실행 방법: `!python classifier.py --use_gpu`
   - 출력 예시:
     ```
     Training Sentiment Classifier on sst...
     ... (학습 로그)
     Evaluating on sst...
     ... (평가 결과)
     Training Sentiment Classifier on cfimdb...
     ... (학습 로그)
     Evaluating on cfimdb...
     ... (평가 결과)
     ```

---

## 4. Part 2 실행 방법

- paraphrase_detection.py: 패러프레이즈 탐지 수행
  - 실행 방법: `!python paraphrase_detection.py --use_gpu`
- sonnet_generation_basic.py: 소네트 생성 수행
  - 실행 방법: `!python sonnet_generation_basic.py --use_gpu`
- sonnet_generation_Lora_Peft.py: Lora 기반 소네트 생성 (확장)
  - 실행 방법: `!python sonnet_generation_Lora_Peft.py --use_gpu`

---

## 5. 기능 확장 TEST

- Sonnet_Lora: `sonnet_generation_Lora_Peft.py` 실행
- 승민이꺼: (관련 파일/설명 추가)
