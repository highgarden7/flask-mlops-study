# Iris Decision Tree API

이 프로젝트는 **Iris 꽃 데이터셋**을 기반으로 학습된 **Decision Tree 모델**을 API로 제공하는 Python Flask 기반 서버입니다.  
`train_model.py`를 실행하여 모델을 학습하고 저장한 뒤, `app.py`를 실행하여 `/predict` 엔드포인트로 예측 요청을 할 수 있습니다.

---

## 📁 프로젝트 구조

```
├── app.py                  # Flask 서버 실행 진입점
├── train_model.py          # Iris 모델 학습 및 저장
├── api/
│   └── predict.py          # 예측 API 로직
├── model/
│   └── iris_model.pkl      # 학습된 모델 파일 (train_model.py 실행 후 생성됨)
└── requirements.txt        # (필요한 경우) 의존성 패키지
```

---

## ⚙️ 설치 및 실행 방법

1. **의존성 설치**

   ```bash
   pip install -r requirements.txt
   ```

   또는 필요한 패키지를 수동 설치:

   ```bash
   pip install scikit-learn flask joblib pandas
   ```

2. **모델 학습 및 저장**

   ```bash
   python train_model.py
   ```

3. **서버 실행**

   ```bash
   python app.py
   ```

4. **API 요청 예시 (POST /predict)**

   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "features": [
         [5.1, 3.5, 1.4, 0.2],
         [6.2, 2.9, 4.3, 1.3]
       ]
     }'
   ```

   **응답 예시:**

   ```json
   {
     "predictions": [
       { "prediction": 0, "label": "setosa" },
       { "prediction": 1, "label": "versicolor" }
     ]
   }
   ```

---

## 🔍 참고

- 사용한 데이터셋: [Iris dataset (scikit-learn)](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- 사용한 모델: `DecisionTreeClassifier` (scikit-learn)
- REST API 서버: Flask
