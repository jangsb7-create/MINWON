import joblib

# 모델과 벡터라이저 로드
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def classify_complaint(text):
    # 텍스트 벡터화
    text_vec = vectorizer.transform([text])
    # 예측
    prediction = model.predict(text_vec)
    return prediction[0]

# 예시 사용
if __name__ == "__main__":
    sample_text = "길에 불법 광고물이 많아요."
    category = classify_complaint(sample_text)
    print(f"민원 내용: {sample_text}")
    print(f"분류된 카테고리: {category}")