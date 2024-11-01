import os
import cv2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

image_folder = r"D:\Workspace\XuLyAnh\LocAnh\image_paths"
label_file_path = r"D:\Workspace\XuLyAnh\LocAnh\labels\labels.csv"

labels_df = pd.read_csv(label_file_path)

def load_images_and_labels(image_folder, labels_df, image_size=(30, 30)):
    images = []
    labels = []
    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        label = row['label']
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, image_size)  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            images.append(image.flatten())  
            labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_images_and_labels(image_folder, labels_df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    return training_time, accuracy, precision, recall

models = {
    'SVM': SVC(kernel='linear'),
    'KNN': KNeighborsClassifier(n_neighbors=1),  
    'Decision Tree': DecisionTreeClassifier(max_depth=5)
}

results = {}
for model_name, model in models.items():
    training_time, accuracy, precision, recall = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = {
        'Time': training_time,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

for model_name, metrics in results.items():
    print(f"{model_name} results:")
    print(f" - Training Time: {metrics['Time']:.4f} seconds")
    print(f" - Accuracy: {metrics['Accuracy']:.4f}")
    print(f" - Precision: {metrics['Precision']:.4f}")
    print(f" - Recall: {metrics['Recall']:.4f}")
    print()


image_size = (300, 300)  

labels_df = pd.read_csv(label_file_path)

labels_dict = dict(zip(labels_df["image_name"], labels_df["label"]))

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

num_images = len(image_files)
num_cols = 4 
num_rows = (num_images + num_cols - 1) // num_cols  

combined_image = np.zeros((image_size[1] * num_rows + 20 * num_rows, image_size[0] * num_cols, 3), dtype=np.uint8)

for idx, image_name in enumerate(image_files):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, image_size)
        label = labels_dict.get(image_name, "unknown")
        cv2.putText(image, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        row = idx // num_cols
        col = idx % num_cols
        y_start = row * (image_size[1] + 20)
        y_end = y_start + image_size[1]
        x_start = col * image_size[0]
        x_end = x_start + image_size[0]
        combined_image[y_start:y_end, x_start:x_end] = image

cv2.imshow("Combined Image with Labels", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
