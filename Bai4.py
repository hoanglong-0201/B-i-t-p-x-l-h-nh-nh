import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm, neighbors, tree
from tensorflow.keras.datasets import cifar10
import time

# Bước 1: Tải dữ liệu và tiền xử lý
# Tải dữ liệu CIFAR-10 (gồm 10 loại: chó, mèo, xe...)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Chuyển đổi hình ảnh 32x32x3 thành vector để làm đầu vào cho mô hình
X_train = X_train.reshape((50000, 32 * 32 * 3))
X_test = X_test.reshape((10000, 32 * 32 * 3))

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bước 2: Định nghĩa hàm đo lường hiệu suất và in kết quả
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Bắt đầu đếm thời gian
    start_time = time.time()
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train.ravel())
    
    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    
    # Đo lường thời gian, độ chính xác, precision và recall
    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    
    # In kết quả
    print(f"Model: {model.__class__.__name__}")
    print(f"Time: {elapsed_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-" * 30)

# Bước 3: Khởi tạo và huấn luyện các mô hình

# SVM
svm_model = svm.SVC()
evaluate_model(svm_model, X_train, y_train, X_test, y_test)

# KNN
knn_model = neighbors.KNeighborsClassifier()
evaluate_model(knn_model, X_train, y_train, X_test, y_test)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()
evaluate_model(dt_model, X_train, y_train, X_test, y_test)
