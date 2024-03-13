from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('data.csv')

# Pisahkan atribut dan label
X = data[['ip1', 'ip2', 'ip3', 'ip4']]
y = data['tepat']

# Bagi data menjadi training set dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Lakukan prediksi pada data test
y_pred = knn_model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)

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
