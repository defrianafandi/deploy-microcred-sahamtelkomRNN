from flask import Flask, request, render_template
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html', predict_close=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Mengambil nilai close dan predict ke model
    '''
    scaler = MinMaxScaler(feature_range=(0,1))
    close = request.form.values()
    close = float(close)
    x = scaler.fit_transform(np.array(close).reshape(-1,1))
    prediction = model.predict(x)
    hasil_predict = scaler.inverse_transform(prediction)
    hasil_predict = hasil_predict[0,0]

    return render_template('index.html', predict_close=hasil_predict)


if __name__ == '__main__':
    app.run(debug=True)