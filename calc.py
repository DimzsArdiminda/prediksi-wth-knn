import pandas as pd
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

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