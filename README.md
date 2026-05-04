# MINWON

민원 자동분류 시스템

## 기능
- 민원 텍스트를 입력받아 자동으로 카테고리를 분류합니다.
- 머신러닝을 사용하여 텍스트 분류를 수행합니다.

## 설치
1. 의존성 설치:
   ```
   pip install -r requirements.txt
   ```

2. 모델 훈련:
   ```
   python train.py
   ```

3. 분류 사용:
   ```
   python classify.py
   ```

## 파일 구조
- `data/complaints.csv`: 샘플 민원 데이터
- `train.py`: 모델 훈련 스크립트
- `classify.py`: 민원 분류 스크립트
- `model.pkl`: 훈련된 모델
- `vectorizer.pkl`: 텍스트 벡터라이저