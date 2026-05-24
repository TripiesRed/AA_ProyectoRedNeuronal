# Student Habits vs. Academic Performance — Classification Project

Proyecto para la asignatura de **Aprendizaje Automático** impartida en la **UCM**.

## Descripción

Este proyecto analiza el impacto de los hábitos de los estudiantes en su rendimiento académico y clasifica sus calificaciones finales en cuatro categorías: *Failed*, *Average*, *Good* y *Excellent*. Se implementan y comparan dos enfoques de Machine Learning construidos desde cero:

- **Red Neuronal Artificial (ANN)** con retropropagación y Early Stopping.
- **SVM con Kernel Gaussiano (RBF)** de scikit-learn.

El flujo de trabajo incluye descarga automática del dataset, preprocesamiento, optimización de hiperparámetros mediante *Grid Search*, evaluación sobre un conjunto de test independiente, reducción de dimensionalidad con **PCA** para visualización y análisis de importancia de características.

---

## 📂 Estructura del Proyecto

```
.
├── main.py            # Pipeline principal: descarga, preprocesado, entrenamiento y evaluación
├── multi_class.py     # One-vs-All y forward propagation / predict de la red neuronal
├── utils.py           # Coste, backprop, bucle de entrenamiento (con Early Stopping) y gráficos
├── logistic_reg.py    # Regresión logística regularizada: sigmoid, coste, gradiente y GD por lotes
└── README.md
```

### Descripción de cada módulo

| Fichero | Responsabilidad |
|---|---|
| `main.py` | Coordina todo el pipeline: descarga del dataset con `kagglehub`, preprocesado, Grid Search de la ANN y del SVM, evaluación final con matrices de confusión y análisis PCA. |
| `multi_class.py` | `oneVsAll` — entrena K clasificadores logísticos (uno por clase). `predictOneVsAll` — predicción por argmax sobre las K puntuaciones. `forward_propagation` — calcula A1, A2 y H para la ANN de dos capas. `predict` — predicción final de la ANN. |
| `utils.py` | `cost` y `backprop` — función de coste y gradientes de la ANN con regularización L2. `train` — bucle de entrenamiento con Early Stopping (paciencia = 500 épocas). `plot_learning_curves`, `plot_decision_boundary`, `plot_pca_analysis` — visualizaciones. |
| `logistic_reg.py` | Bloque fundacional: `sigmoid`, `compute_cost`, `compute_gradient`, `compute_cost_reg`, `compute_gradient_reg` y `gradient_descent` por lotes. Usado por `multi_class.py`. |

---

## 🗂️ Dataset

- **Fuente:** [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance) (Kaggle).
- **Descarga:** automática mediante `kagglehub` al ejecutar `main.py`.
- **Variable objetivo:** `exam_score` discretizada en 4 clases con los siguientes intervalos:

| Clase | Rango de puntuación |
|---|---|
| Failed | [0, 50] |
| Average | (50, 70] |
| Good | (70, 90] |
| Excellent | (90, 100] |

---

## ⚙️ Preprocesado

1. **Valores nulos:** los `NaN` en `parental_education_level` se interpretan como nivel básico (`"Basic school"`).
2. **Duplicados e IDs:** se eliminan filas duplicadas y la columna `student_id`.
3. **Codificación de variables:**
   - Binarización de `part_time_job` y `extracurricular_participation` (Yes/No → 1/0).
   - *One-hot encoding* de `gender`, `diet_quality`, `parental_education_level` e `internet_quality` (con `drop_first=True`).
4. **División del dataset** (estratificada por clase):
   - Train: 70 % · Validation: 15 % · Test: 15 %
5. **Normalización:** `StandardScaler` ajustado **solo** sobre el conjunto de entrenamiento y aplicado al resto. Las columnas binarias y *dummies* se excluyen del escalado.

---

## 🤖 Modelos

### Red Neuronal Artificial (ANN)

Arquitectura de **dos capas** (una oculta + salida):

- **Entrada:** número de características tras el preprocesado.
- **Capa oculta:** neuronas con activación sigmoide (tamaño configurable).
- **Capa de salida:** 4 neuronas (una por clase) con activación sigmoide.
- **Inicialización de pesos:** uniforme en `[-0.12, 0.12]`.
- **Función de coste:** entropía cruzada binaria con regularización L2.
- **Optimización:** Gradient Descent por lotes.
- **Early Stopping:** paciencia de 500 épocas sin mejora en la pérdida de validación (umbral `1e-4`).

**Grid Search de hiperparámetros:**

| Hiperparámetro | Valores explorados |
|---|---|
| Learning rate | 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 |
| Regularización λ | 0.001, 0.01, 0.1, 1.0 |
| Tamaño capa oculta | 25, 50 |
| Épocas máximas | 5 000 |

### SVM con Kernel RBF (Gaussiano)

- Implementado con `sklearn.svm.SVC`.
- Parámetro de kernel: `gamma = 1 / (2 * sigma²)`.
- **Grid Search** sobre C y sigma, ambos en `{0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}` (64 combinaciones).
- El modelo óptimo se selecciona por accuracy en el conjunto de validación.
- Para la **visualización** de la frontera de decisión, el SVM se re-entrena sobre las 2 primeras componentes principales (PCA 2D).

---

## 📊 Evaluación

Ambos modelos se evalúan sobre el **conjunto de test** (nunca visto durante el entrenamiento ni la selección de hiperparámetros):

- **Accuracy** global.
- **Matrices de confusión** para las 4 clases.

---

## 🔍 Análisis PCA

Tras la evaluación se realiza un análisis de componentes principales sobre el conjunto de entrenamiento:

- Gráfico de varianza explicada individual (IEV) y acumulada (CEV).
- *Scree plot* con criterio de Kaiser (eigenvalue ≥ 1).
- Contribución global de cada característica original a las 8 primeras componentes principales.

---

## 🛠️ Requisitos e Instalación

Python 3.x. Instala las dependencias con:

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib
```

> **Nota:** para que `kagglehub` pueda descargar el dataset es necesario tener configuradas las credenciales de Kaggle (`~/.kaggle/kaggle.json` o las variables de entorno `KAGGLE_USERNAME` y `KAGGLE_KEY`).

---

## ▶️ Ejecución

```bash
python main.py
```

El script ejecuta automáticamente todo el pipeline: descarga, preprocesado, Grid Search de ambos modelos, evaluación y visualizaciones.