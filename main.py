import kagglehub
import pandas as pd
import os
import utils 
import numpy as np

def main():

   #####################################################
   # 0) DESCARGA DEL DATASET 

   # Descarga la última versión 
   path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")
   print("Path to dataset files:", path)

   # Lista los archivos descargados
   archivos = os.listdir(path)
   print(archivos)

   # Carga el dataset
   dataset = pd.read_csv(os.path.join(path, archivos[0]))
   dataset.head()
   print("Dataset shape:", dataset.shape)

   #####################################################
   # 1) PREPROCESADO

   # Comprobación de características con datos faltantes
   print(dataset.isnull().sum())
   # NOTA: En este caso detecta "None" como valor nulo, aunque podría también entenderse como
   # una propia categoría de nivel de estudios "Ninguno" para una persona que no ha asistido a la 
   # escuela. Dado que no termina de estar claro,
   # vamos a interpretar "None" como indicativo de que el nivel de estudios alcanzado es el más
   # básico posible, que llamaremos "Basic school"
   dataset["parental_education_level"] = dataset["parental_education_level"].fillna("Basic school")
   print(dataset.isnull().sum())

   # Eliminamos muestras duplicadas y la primera columna (son ids sin utilidad)
   dataset = dataset.drop_duplicates()
   dataset = dataset.drop(columns=["student_id"])
   print(f"Clean Dataset: {dataset.shape}")

   # Discretización variable objetivo (exam_score)
   bins = [0.0, 50.0, 70.0, 90.0, 100.0]
   labels = [ "Failed", "Average", "Good", "Excellent"]
   dataset["exam_score_cat"] = pd.cut(dataset["exam_score"], bins=bins, labels=labels, include_lowest=True)
   print(dataset["exam_score_cat"].value_counts())

   # Definimos las variables
   X_raw = dataset.drop(columns=["exam_score", "exam_score_cat"])
   y_raw = dataset["exam_score_cat"]

   # Transformación de las características no numéricas
   # Binarización
   X_raw["gender"] = X_raw["gender"].map({"Male": 2, "Female": 1, "Other": 0})
   X_raw["part_time_job"] = X_raw["part_time_job"].map({"Yes": 1, "No":0})
   X_raw["extracurricular_participation"] = X_raw["extracurricular_participation"].map({"Yes": 1, "No":0})
   # One-hot encoding
   X = pd.get_dummies(X_raw, columns=["diet_quality","parental_education_level","internet_quality"], 
                      drop_first=True)
   X = X.astype(float) # Aseguramos mantener X numérico
   # Codificación para el target
   target_map = {
    "Failed": 0,
    "Average": 1,
    "Good": 2,
    "Excellent": 3
   }
   # Aplicamos el mapeo
   y = y_raw.map(target_map)

   # División en train-validation-test
   from sklearn.model_selection import train_test_split
   # Separacion Train y Validation-Test (70-30)
   X_train, X_tmp, y_train, y_tmp = train_test_split(
      X, y,test_size=0.3, random_state=42, stratify=y  
   )
   # Separación Validation y test (70-15-15)
   X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42,stratify=y_tmp
   )

   # Normalización de las características
   from sklearn.preprocessing import StandardScaler, MinMaxScaler
   # Dejamos fuera las columnas binarias/dummy para no alterar su codificación de 0 y 1
   columnas_no_escalar = ["gender", "part_time_job", "extracurricular_participation"]
   # Excluimos también las columnas generadas por get_dummies
   columnas_numericas = [col for col in X.columns if col not in columnas_no_escalar 
                        and not any(pref in col for pref in ["diet_quality_", "parental_education_", "internet_quality_"])]
   scaler = StandardScaler()
   #scaler = MinMaxScaler()
   X_train_scaled = X_train.copy()
   X_train_scaled[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])

   # Normalizamos los conjuntos de validación y test en base al fit realizado en el train-set
   X_val_scaled = X_val.copy()
   X_val_scaled[columnas_numericas] = scaler.transform(X_val[columnas_numericas])
   X_test_scaled = X_test.copy()
   X_test_scaled[columnas_numericas] = scaler.transform(X_test[columnas_numericas])

   # Comprobamos que se realizó la estratificación
   compare = pd.DataFrame({
      "Train":      np.unique(y_train, return_counts=True),
      "Validation": np.unique(y_val, return_counts=True),
      "Test":       np.unique(y_test, return_counts=True)
   }).round(3)

   print(compare)

   #####################################################
   # 2) MODELOS

   # Red Neuronal entrenada con Gradient Descent
   # NOTA: He visto una pequeña mejora usando ".values" respecto a ".to_numpy()" de hasta un 0.6%.
   # Obtamos por usar to_numpy() por ser el estándar actual que hemos visto en clase
   X_train_scaled = X_train_scaled.to_numpy()
   X_val_scaled = X_val_scaled.to_numpy()
   y_train_np = pd.get_dummies(y_train).astype(float).to_numpy()
   y_val_np = pd.get_dummies(y_val).astype(float).to_numpy()

   thetas_1 = []
   thetas_2 = []
   train_loss_v, val_loss_v = [] val_loss_v
    = []
   accuracies = []
   # Entrenamiento
   theta1, theta2, train_losses, val_losses, acc = utils.train(
      X_train_scaled, y_train_np, X_val_scaled, y_val_np, hidden_size= 25,
      learning_rate = 0.5, lambda_= 0.01, epochs= 1000
   )

   # Curvas de aprendizaje
   #utils.plot_learning_curves(train_losses, val_losses)










if __name__ == "__main__":
   main()