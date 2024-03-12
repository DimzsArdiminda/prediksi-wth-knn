from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

# Inisialisasi aplikasi Flask
app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        ip1 = float(request.form['ip1'])
        ip2 = float(request.form['ip2'])
        ip3 = float(request.form['ip3'])
        ip4 = float(request.form['ip4'])
        
        # akurasi
        prediction = knn_model.predict([[ip1, ip2, ip3, ip4]])

        # Hitung akurasi model
        accuracy = knn_model.score(X_test, y_test)

    return render_template('index.html', prediction=prediction[0], accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
