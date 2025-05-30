from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
import os

print("🔍 Step 1: Iris 데이터 로딩 중...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("✅ Iris 데이터 로딩 완료!")

print("\n📊 Step 2: 데이터 분할 중 (train/test)...")
X = df[iris.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ 데이터 분할 완료!")

print("\n🌳 Step 3: Decision Tree 모델 학습 중...")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("✅ 모델 학습 완료!")

print("\n💾 Step 4: 모델 저장 중...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/iris_model.pkl")
print("✅ 모델이 'model/iris_model.pkl' 로 저장되었습니다.")

# 성능 출력
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\n📈 학습 정확도 (Train): {train_score:.2f}")
print(f"📈 테스트 정확도 (Test): {test_score:.2f}")
