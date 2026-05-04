import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 데이터 로드
data = pd.read_csv('data/complaints.csv')

# 텍스트와 라벨 분리
X = data['text']
y = data['category']

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 벡터화
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 모델 훈련
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 평가
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 모델과 벡터라이저 저장
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')