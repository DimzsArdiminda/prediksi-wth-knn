from flask import Flask, render_template, request
from calc import accuracy, knn_model

# Inisialisasi aplikasi Flask
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy)

@app.route('/', methods=['POST'])
def predict():
    ip1 = float(request.form['ip1'])
    ip2 = float(request.form['ip2'])
    ip3 = float(request.form['ip3'])
    ip4 = float(request.form['ip4'])

    # Lakukan prediksi
    prediction = knn_model.predict([[ip1, ip2, ip3, ip4]])

    return render_template('index.html', prediction=prediction[0], accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
