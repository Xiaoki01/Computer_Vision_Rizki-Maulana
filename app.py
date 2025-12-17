from flask import Flask, render_template, request
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from math import log2
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset dan train model KNN sekali saja saat aplikasi dimulai
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# ========================
# NAIVE BAYES FUNCTIONS
# ========================

def create_credit_dataset():
    """Membuat dataset kelayakan kredit"""
    data_list = [
        ("Muda", "Rendah", "Ya", "Tidak"),
        ("Muda", "Rendah", "Tidak", "Tidak"),
        ("Muda", "Sedang", "Tidak", "Ya"),
        ("Tengah", "Rendah", "Ya", "Ya"),
        ("Tengah", "Sedang", "Tidak", "Ya"),
        ("Tengah", "Tinggi", "Ya", "Ya"),
        ("Tua", "Sedang", "Tidak", "Tidak"),
        ("Tua", "Tinggi", "Tidak", "Ya"),
        ("Muda", "Tinggi", "Ya", "Ya"),
        ("Tua", "Rendah", "Tidak", "Tidak"),
        ("Tengah", "Tinggi", "Tidak", "Ya"),
        ("Muda", "Sedang", "Ya", "Ya"),
    ]
    
    data = {
        'Usia': [row[0] for row in data_list],
        'Pendapatan': [row[1] for row in data_list],
        'Punya_Rumah': [row[2] for row in data_list],
        'Layak_Kredit': [row[3] for row in data_list]
    }
    return pd.DataFrame(data)

def calculate_prior_probabilities(df, target_col='Layak_Kredit'):
    """Menghitung probabilitas prior P(Layak) dan P(Tidak Layak)"""
    value_counts = df[target_col].value_counts()
    total = len(df)
    priors = {}
    for val, count in value_counts.items():
        priors[val] = count / total
    return priors

def calculate_likelihood(df, feature, feature_value, target, target_value):
    """Menghitung likelihood P(Feature=value|Target=value)"""
    subset = df[df[target] == target_value]
    if len(subset) == 0:
        return 0
    feature_count = len(subset[subset[feature] == feature_value])
    return feature_count / len(subset)

def calculate_all_likelihoods(df, features, target='Layak_Kredit'):
    """Menghitung semua likelihood untuk setiap kombinasi fitur dan target"""
    likelihoods = {}
    target_values = df[target].unique()
    
    for feature in features:
        likelihoods[feature] = {}
        feature_values = df[feature].unique()
        for fval in feature_values:
            likelihoods[feature][fval] = {}
            for tval in target_values:
                likelihood = calculate_likelihood(df, feature, fval, target, tval)
                likelihoods[feature][fval][tval] = likelihood
    
    return likelihoods

def manual_naive_bayes_predict(priors, likelihoods, features_dict, target_values):
    """Prediksi manual menggunakan Naive Bayes"""
    posteriors = {}
    
    for target_val in target_values:
        posterior = priors[target_val]
        
        for feature, feature_val in features_dict.items():
            if feature in likelihoods and feature_val in likelihoods[feature]:
                posterior *= likelihoods[feature][feature_val].get(target_val, 0)
        
        posteriors[target_val] = posterior
    
    # Normalisasi
    total = sum(posteriors.values())
    if total > 0:
        posteriors = {k: v/total for k, v in posteriors.items()}
    
    # Pilih kelas dengan probabilitas tertinggi
    prediction = max(posteriors, key=posteriors.get)
    
    return prediction, posteriors

def train_sklearn_naive_bayes(df, features, target='Layak_Kredit'):
    """Melatih model Naive Bayes menggunakan scikit-learn"""
    X = df[features]
    y = df[target]
    
    # Encode fitur kategori
    encoders = {}
    X_encoded = pd.DataFrame()
    
    for col in features:
        encoders[col] = LabelEncoder()
        X_encoded[col] = encoders[col].fit_transform(X[col])
    
    # Encode target
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    # Train model
    model = CategoricalNB()
    model.fit(X_encoded.values, y_encoded)
    
    return model, encoders, y_encoder

def create_probability_chart(posteriors):
    """Membuat visualisasi probabilitas posterior"""
    labels = list(posteriors.keys())
    values = list(posteriors.values())
    colors = ['#28a745' if label == 'Ya' else '#dc3545' for label in labels]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Probabilitas Posterior', fontsize=12, fontweight='bold')
    ax.set_xlabel('Kelayakan Kredit', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Probabilitas Posterior', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

# ========================
# DECISION TREE FUNCTIONS
# ========================

def create_golf_dataset():
    """Membuat dataset Play Golf"""
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny',
                    'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild',
                        'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High',
                     'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': ['False', 'False', 'True', 'False', 'False', 'True', 'True', 'False',
                  'False', 'False', 'True', 'True', 'False', 'True'],
        'PlayGolf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                     'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    return pd.DataFrame(data)

def entropy(target_col):
    """Menghitung Entropy"""
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    for i in range(len(elements)):
        probability = counts[i] / np.sum(counts)
        if probability > 0:
            entropy_value -= probability * log2(probability)
    return round(entropy_value, 4)

def info_gain(data, split_attribute_name, target_name="PlayGolf"):
    """Menghitung Information Gain"""
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target_name])

    information_gain = total_entropy - weighted_entropy
    return round(information_gain, 4)

def calculate_all_gains(data, features, target_name="PlayGolf"):
    """Menghitung Information Gain untuk semua fitur"""
    gains = {}
    for feature in features:
        gains[feature] = info_gain(data, feature, target_name)
    return gains

def Id3(data, originaldata, features, target_attribute_name="PlayGolf", parent_node_class=None):
    """Algoritma ID3 untuk membentuk Decision Tree"""
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])
        ]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        ]

        gains = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(gains)
        best_feature = features[best_feature_index]

        tree_structure = {best_feature: {}}

        remaining_features = [f for f in features if f != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = Id3(sub_data, data, remaining_features, target_attribute_name, parent_node_class)
            tree_structure[best_feature][value] = subtree

    return tree_structure

def create_tree_visualization(df_encoded, X, y):
    """Membuat visualisasi Decision Tree menggunakan sklearn"""
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf = clf.fit(X, y)
    
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, feature_names=list(X.columns), class_names=['No', 'Yes'], 
                   filled=True, rounded=True, fontsize=10)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def predict_golf(tree_dict, outlook, temperature, humidity, windy):
    """Prediksi menggunakan Decision Tree yang sudah dibuat"""
    def traverse_tree(tree, attributes):
        if not isinstance(tree, dict):
            return tree
        
        root = list(tree.keys())[0]
        value = attributes.get(root)
        
        if value in tree[root]:
            subtree = tree[root][value]
            return traverse_tree(subtree, attributes)
        else:
            return "Unknown"
    
    attributes = {
        'Outlook': outlook,
        'Temperature': temperature,
        'Humidity': humidity,
        'Windy': windy
    }
    
    return traverse_tree(tree_dict, attributes)

# ========================
# GLCM FUNCTIONS
# ========================

def extract_glcm_features(image_path, degree):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if degree == 0:
        distances = [1]
        angles = [0]
    elif degree == 45:
        distances = [1]
        angles = [np.pi/4]
    elif degree == 90:
        distances = [1]
        angles = [np.pi/2]
    else:  # 135
        distances = [1]
        angles = [3*np.pi/4]
    
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return {
        'contrast': contrast,
        'energy': energy,
        'homogeneity': homogeneity,
        'correlation': correlation
    }

# ========================
# KNN FUNCTIONS
# ========================

def preprocess_uploaded_digit(image_path):
    """Preprocess gambar dengan metode yang lebih baik - kompresi penuh tanpa cropping"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary
    
    max_dim = max(cropped.shape[0], cropped.shape[1])
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    y_offset = (max_dim - cropped.shape[0]) // 2
    x_offset = (max_dim - cropped.shape[1]) // 2
    square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
    
    img_resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
    
    img_normalized = (img_resized / 255.0 * 16).astype(np.float64)
    
    return img_normalized.reshape(1, -1)

def create_visualization(num_samples=5):
    """Membuat visualisasi hasil prediksi"""
    y_pred = knn_model.predict(X_test)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'Label: {y_test[i]}\nPred: {y_pred[i]}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_confusion_matrix_plot(y_true, y_pred):
    """Membuat visualisasi confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=range(10), yticklabels=range(10),
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

# ========================
# ROUTES
# ========================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tugas1', methods=['GET', 'POST'])
def tugas1():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('tugas1.html')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('tugas1.html')
        
        degree = int(request.form.get('degree', 0))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = extract_glcm_features(filepath, degree)
        image_path = f'uploads/{filename}'
        
        return render_template('tugas1.html', result=result, degree=degree, image_path=image_path)
    
    return render_template('tugas1.html')

@app.route('/tugas2', methods=['GET', 'POST'])
def tugas2():
    active_tab = 'evaluate'
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'evaluate':
            active_tab = 'evaluate'
            y_pred = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            viz_img = create_visualization()
            cm_img = create_confusion_matrix_plot(y_test, y_pred)
            
            return render_template('tugas2.html', 
                                 accuracy=accuracy,
                                 report=report,
                                 visualization=viz_img,
                                 confusion_matrix=cm_img,
                                 evaluated=True,
                                 active_tab=active_tab)
        
        elif action == 'predict' and 'image' in request.files:
            active_tab = 'predict'
            file = request.files['image']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                processed_img = preprocess_uploaded_digit(filepath)
                prediction = knn_model.predict(processed_img)[0]
                probabilities = knn_model.predict_proba(processed_img)[0]
                
                img_8x8 = processed_img.reshape(8, 8)
                plt.figure(figsize=(3, 3))
                plt.imshow(img_8x8, cmap='gray_r')
                plt.title(f'Preprocessed (8x8)', fontsize=10)
                plt.axis('off')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                processed_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
                image_path = f'uploads/{filename}'
                
                return render_template('tugas2.html',
                                     prediction=prediction,
                                     probabilities=probabilities,
                                     image_path=image_path,
                                     processed_image=processed_base64,
                                     predicted=True,
                                     active_tab=active_tab)
    
    return render_template('tugas2.html', active_tab=active_tab)

@app.route('/tugas3', methods=['GET', 'POST'])
def tugas3():
    # Buat dataset
    df = create_golf_dataset()
    
    # Hitung entropy awal
    initial_entropy = entropy(df['PlayGolf'])
    
    # Hitung information gain untuk semua fitur
    features = list(df.columns[:-1])
    all_gains = calculate_all_gains(df, features)
    
    # Buat decision tree manual dengan ID3
    decision_tree = Id3(df, df, features)
    
    # Encode data untuk sklearn
    le = LabelEncoder()
    df_encoded = df.copy()
    for col in df.columns:
        df_encoded[col] = le.fit_transform(df[col])
    
    X_golf = df_encoded.drop(columns=['PlayGolf'])
    y_golf = df_encoded['PlayGolf']
    
    # Buat visualisasi tree
    tree_image = create_tree_visualization(df_encoded, X_golf, y_golf)
    
    prediction_result = None
    
    if request.method == 'POST':
        outlook = request.form.get('outlook')
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')
        windy = request.form.get('windy')
        
        prediction = predict_golf(decision_tree, outlook, temperature, humidity, windy)
        prediction_result = {
            'outlook': outlook,
            'temperature': temperature,
            'humidity': humidity,
            'windy': windy,
            'result': prediction
        }
    
    return render_template('tugas3.html',
                         dataset=df.to_html(classes='table table-striped table-hover', index=False),
                         initial_entropy=initial_entropy,
                         information_gains=all_gains,
                         decision_tree=decision_tree,
                         tree_visualization=tree_image,
                         prediction=prediction_result)

@app.route('/tugas4', methods=['GET', 'POST'])
def tugas4():
    # Buat dataset
    df = create_credit_dataset()
    
    # Fitur dan target
    features = ['Usia', 'Pendapatan', 'Punya_Rumah']
    target = 'Layak_Kredit'
    
    # Hitung probabilitas prior
    priors = calculate_prior_probabilities(df, target)
    
    # Hitung semua likelihood
    likelihoods = calculate_all_likelihoods(df, features, target)
    
    # Train sklearn model
    sklearn_model, encoders, y_encoder = train_sklearn_naive_bayes(df, features, target)
    
    # Evaluasi model dengan data training
    X_eval = df[features]
    X_encoded = pd.DataFrame()
    for col in features:
        X_encoded[col] = encoders[col].transform(X_eval[col])
    
    y_true = df[target]
    y_pred_encoded = sklearn_model.predict(X_encoded.values)
    y_pred = y_encoder.inverse_transform(y_pred_encoded)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    prediction_result = None
    probability_chart = None
    calculation_steps = None
    
    if request.method == 'POST':
        usia = request.form.get('usia')
        pendapatan = request.form.get('pendapatan')
        punya_rumah = request.form.get('punya_rumah')
        
        # Prediksi manual
        features_dict = {
            'Usia': usia,
            'Pendapatan': pendapatan,
            'Punya_Rumah': punya_rumah
        }
        
        target_values = df[target].unique()
        manual_prediction, posteriors = manual_naive_bayes_predict(
            priors, likelihoods, features_dict, target_values
        )
        
        # Prediksi dengan sklearn
        test_data = pd.DataFrame({
            'Usia': [usia],
            'Pendapatan': [pendapatan],
            'Punya_Rumah': [punya_rumah]
        })
        
        test_encoded = pd.DataFrame()
        for col in features:
            test_encoded[col] = encoders[col].transform(test_data[col])
        
        sklearn_prediction_encoded = sklearn_model.predict(test_encoded.values)[0]
        sklearn_prediction = y_encoder.inverse_transform([sklearn_prediction_encoded])[0]
        
        sklearn_proba = sklearn_model.predict_proba(test_encoded.values)[0]
        sklearn_posteriors = dict(zip(y_encoder.classes_, sklearn_proba))
        
        # Buat visualisasi
        probability_chart = create_probability_chart(posteriors)
        
        # Langkah perhitungan manual
        calculation_steps = {
            'priors': priors,
            'likelihoods': {},
            'posteriors': posteriors
        }
        
        for feature, value in features_dict.items():
            if feature in likelihoods and value in likelihoods[feature]:
                calculation_steps['likelihoods'][f'{feature}={value}'] = likelihoods[feature][value]
        
        prediction_result = {
            'usia': usia,
            'pendapatan': pendapatan,
            'punya_rumah': punya_rumah,
            'manual_prediction': manual_prediction,
            'sklearn_prediction': sklearn_prediction,
            'manual_posteriors': posteriors,
            'sklearn_posteriors': sklearn_posteriors
        }
    
    return render_template('tugas4.html',
                         dataset=df.to_html(classes='table table-striped table-hover', index=False),
                         priors=priors,
                         likelihoods=likelihoods,
                         accuracy=accuracy,
                         prediction=prediction_result,
                         probability_chart=probability_chart,
                         calculation_steps=calculation_steps)

if __name__ == '__main__':
    app.run(debug=True)