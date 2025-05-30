from flask import Flask
from api.predict import predict

app = Flask(__name__)

# 라우트 설정
app.add_url_rule('/predict', view_func=predict, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
