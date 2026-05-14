"""
train_models.py — Script para re-entrenar los modelos
Ejecutar: python train_models.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib, json, os

os.makedirs('models', exist_ok=True)

print("📂 Cargando dataset...")
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

print(f"   Registros: {len(df)}, Clases: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n📊 Entrenando Regresión Logística...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
print(f"   Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['Sin Enfermedad','Con Enfermedad']))

print("\n🧠 Entrenando Red Neuronal (MLP)...")
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
nn.fit(X_train_s, y_train)
y_pred_nn = nn.predict(X_test_s)
print(f"   Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(classification_report(y_test, y_pred_nn, target_names=['Sin Enfermedad','Con Enfermedad']))

joblib.dump(lr, 'models/logistic_regression.pkl')
joblib.dump(nn, 'models/neural_network.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

def get_metrics(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'name': name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'confusion_matrix': cm,
        'precision_0': round(report['0']['precision'], 4),
        'recall_0': round(report['0']['recall'], 4),
        'f1_0': round(report['0']['f1-score'], 4),
        'precision_1': round(report['1']['precision'], 4),
        'recall_1': round(report['1']['recall'], 4),
        'f1_1': round(report['1']['f1-score'], 4),
    }

metrics = {
    'logistic': get_metrics(y_test, y_pred_lr, 'Regresión Logística'),
    'neural': get_metrics(y_test, y_pred_nn, 'Red Neuronal'),
    'feature_names': list(X.columns),
    'classes': ['Sin Enfermedad', 'Con Enfermedad']
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Modelos guardados en models/")
print(f"   LR Accuracy:  {metrics['logistic']['accuracy']:.4f}")
print(f"   NN Accuracy:  {metrics['neural']['accuracy']:.4f}")
