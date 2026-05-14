# CardioPredict 🫀

**Proyecto Final Individual — Entrega: 11-MAY-2026**

Aplicación web de clasificación de enfermedades cardíacas usando dos modelos de Machine Learning:
- **Regresión Logística** (~95% accuracy)
- **Red Neuronal MLP** (~92% accuracy)

Dataset: UCI Heart Disease (archive.ics.uci.edu) — 303 registros, 13 características.

---

## 📁 Estructura del Proyecto

```
proyecto_final/
├── app.py                    # Servidor Flask (backend)
├── heart.csv                 # Dataset UCI Heart Disease
├── models/
│   ├── logistic_regression.pkl   # Modelo LR entrenado
│   ├── neural_network.pkl        # Modelo MLP entrenado
│   ├── scaler.pkl                # StandardScaler
│   └── metrics.json              # Métricas de evaluación
├── templates/
│   └── index.html            # Página web principal
├── train_models.py           # Script de entrenamiento
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

---

## 🚀 Instalación y Ejecución Local

### Requisitos
- Python 3.8+
- pip

### Pasos

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Re-entrenar modelos
python train_models.py

# 3. Ejecutar el servidor
python app.py

# 4. Abrir en el navegador
# http://localhost:5000
```

---

## 🌐 Despliegue en GitHub Pages (versión estática)

Como Flask requiere backend, para GitHub Pages se recomienda:

### Opción A: GitHub Pages + Backend gratuito (Render.com)

1. **Subir el repositorio a GitHub:**
   ```bash
   git init
   git add .
   git commit -m "CardioPredict - Proyecto Final"
   git remote add origin https://github.com/TU_USUARIO/cardiopredict.git
   git push -u origin main
   ```

2. **Desplegar backend en Render.com (gratis):**
   - Ir a https://render.com
   - New → Web Service → conectar repositorio GitHub
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Obtendrás una URL tipo `https://cardiopredict.onrender.com`

3. **Activar GitHub Pages:**
   - Settings → Pages → Branch: main → /root
   - La página estará en `https://TU_USUARIO.github.io/cardiopredict/`

### Opción B: Railway.app (más sencillo)

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login y deploy
railway login
railway init
railway up
```

---

## 📊 Características de la Aplicación

### 1. Predicción Individual
- Formulario con los 13 campos del dataset
- Selección de modelo (LR o NN)
- Muestra predicción + probabilidades
- Ejemplos de paciente con/sin riesgo

### 2. Predicción por Lotes
- Carga de archivo CSV
- Soporte para drag & drop
- Matriz de confusión interactiva
- Métricas: Accuracy, Precision, Recall, F1
- Tabla de resultados fila por fila

### 3. Información de Modelos
- Parámetros de entrenamiento
- Métricas en test set
- Matrices de confusión comparativas

---

## 🧠 Modelos

### Regresión Logística
```python
LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
```
- Normalización: StandardScaler
- Train/Test split: 80/20

### Red Neuronal (MLP)
```python
MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
              solver='adam', max_iter=500, early_stopping=True)
```
- Arquitectura: 13 → 64 → 32 → 2
- Normalización: StandardScaler

---

## 📋 Variables del Dataset

| Variable | Descripción |
|----------|-------------|
| age | Edad del paciente |
| sex | Sexo (1=M, 0=F) |
| cp | Tipo de dolor torácico (0-3) |
| trestbps | Presión arterial en reposo (mmHg) |
| chol | Colesterol sérico (mg/dl) |
| fbs | Glucosa ayuno >120 mg/dl (1=Sí) |
| restecg | Resultados ECG reposo (0-2) |
| thalach | Frecuencia cardíaca máxima |
| exang | Angina inducida por ejercicio |
| oldpeak | Depresión ST por ejercicio |
| slope | Pendiente ST máximo (0-2) |
| ca | Vasos principales coloreados (0-3) |
| thal | Talasemia (1=Normal, 2=Fijo, 3=Reversible) |
| **target** | **0=Sin enfermedad, 1=Con enfermedad** |

---

## 👨‍💻 Autor
Proyecto Final — Machine Learning  
Dataset: UCI Machine Learning Repository (archive.ics.uci.edu)
