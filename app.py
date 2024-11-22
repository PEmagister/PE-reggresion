from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('static/ClassifiedClient.pkl')

@app.route('/classify', methods=['POST'])
def classify_user():
    data = request.json
    technical = data.get('technical')
    humanitarian = data.get('humanitarian')
    artistic = data.get('artistic')

    # Передбачення групи
    prediction = model.predict([[technical, humanitarian, artistic]])[0]

    # Повертаємо результат як JSON

    return jsonify({'group': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
