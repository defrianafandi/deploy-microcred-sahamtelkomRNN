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
    close = request.form.values()
    prediction = model.predict(float(close))

    return render_template('index.html', predict_close=prediction)


if __name__ == '__main__':
    app.run(debug=True)