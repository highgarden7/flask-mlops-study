from flask import request, jsonify
import joblib
import numpy as np
import os

model_path = "model/iris_model.pkl"
target_names = ["setosa", "versicolor", "virginica"]

print("📦 모델 로딩 시도 중...")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ 모델 로딩 완료!")
else:
    print("❌ 모델 파일이 없습니다. 먼저 train_model.py를 실행해주세요.")
    model = None

def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()

        # features 키 확인
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features_list = np.array(data['features'])  # (N, 4) 형태

        # 입력 검증
        if features_list.ndim != 2 or features_list.shape[1] != 4:
            return jsonify({"error": "Each input must be a list of 4 numbers."}), 400

        predictions = model.predict(features_list)

        result = [
            {
                "prediction": int(p),
                "label": target_names[int(p)]
            }
            for p in predictions
        ]

        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
