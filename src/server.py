import numpy as np
import xgboost as xgb
from flask import Flask, send_from_directory, request, jsonify
import os

import os
print(os.getcwd())  # Shows current working directory

app = Flask(__name__, static_folder='public')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # TODO: Add your prediction logic here

    print(data['dict'])
    xgb_model = xgb.Booster()
    xgb_model.load_model('models/xgb_koi_model.json')

    # Feature order from exploring_koi.ipynb
    feature_order = [
        'period', 'duration', 'depth', 'prad', 'teq', 'insol', 'model_snr', 'steff', 'slogg', 'srad'
    ]
    
    input_dict = data['dict']

    log_feature_names = list(map(lambda f: 'koi_' + f + '_log', feature_order))

    input_array = np.array([[np.log1p(input_dict[feat]) for feat in feature_order]], dtype=np.float32)
    xgb_input = xgb.DMatrix(input_array, feature_names=log_feature_names)

    prediction = xgb_model.predict(xgb_input)

    label = 'CONFIRMED' if prediction[0] > 0.5 else 'FALSE POSITIVE'
    result = {'prediction': label}
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
