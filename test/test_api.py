import requests

url = "http://127.0.0.1:5000/predict"

sample_input = {
    "features": [
        [5.1, 3.5, 1.4, 0.2],   # setosa
        [6.2, 2.8, 4.8, 1.8],   # virginica
        [5.9, 3.0, 5.1, 1.8]    # virginica
    ]
}

print("🚀 예측 요청 보내는 중...")
response = requests.post(url, json=sample_input)

if response.status_code == 200:
    result = response.json()
    print("✅ 예측 결과:")
    for i, r in enumerate(result['predictions']):
        print(f"  [{i}] 예측: {r['label']} (클래스 번호: {r['prediction']})")
else:
    print("❌ 오류 발생:", response.status_code)
    print("응답 내용:", response.text)
