import kagglehub
import pandas as pd
import os
import utils 
import numpy as np
import matplotlib.pyplot as plt
import multi_class as mc

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
   print("\nDetection of null values in the dataset.")
   print(dataset.isnull().sum())
   # NOTA: En este caso detecta "None" como valor nulo, aunque podría también entenderse como
   # una propia categoría de nivel de estudios "Ninguno" para una persona que no ha asistido a la 
   # escuela. Dado que no termina de estar claro,
   # vamos a interpretar "None" como indicativo de que el nivel de estudios alcanzado es el más
   # básico posible, que llamaremos "Basic school"
   print("\nSolving problem of null values.")
   dataset["parental_education_level"] = dataset["parental_education_level"].fillna("Basic school")
   print(dataset.isnull().sum())

   # Eliminamos muestras duplicadas y la primera columna (son ids sin utilidad)
   dataset = dataset.drop_duplicates()
   dataset = dataset.drop(columns=["student_id"])
   print(f"\nClean Dataset shape: {dataset.shape}")

   # Discretización variable objetivo (exam_score)
   bins = [0.0, 50.0, 70.0, 90.0, 100.0]
   labels = [ "Failed", "Average", "Good", "Excellent"]
   dataset["exam_score_cat"] = pd.cut(dataset["exam_score"], bins=bins, labels=labels, include_lowest=True)
   #print(dataset["exam_score_cat"].value_counts())

   # Definimos las variables
   X_raw = dataset.drop(columns=["age","exam_score", "exam_score_cat"])
   y_raw = dataset["exam_score_cat"]

   # Transformación de las características no numéricas
   # Binarización
   X_raw["part_time_job"] = X_raw["part_time_job"].map({"Yes": 1, "No":0})
   X_raw["extracurricular_participation"] = X_raw["extracurricular_participation"].map({"Yes": 1, "No":0})
   # One-hot encoding
   X = pd.get_dummies(X_raw, columns=["gender","diet_quality","parental_education_level","internet_quality"], 
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
                        and not any(pref in col for pref in ["gender_","diet_quality_", "parental_education_", "internet_quality_"])]
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
   print("\nClass frequency in each set:")
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

   thetas_1, thetas_2 = [] , []
   train_loss_v, val_loss_v = [], []
   accuracies = []
   params = []
   learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
   lambdas = [0.001, 0.01, 0.1, 1.0]
   hidden_sizes = [25, 50]
   
   # Grid Search de los hiperparámetros
   print("\nNeural Network training:\n ")
   for lr in learning_rates:
      for lmbd in lambdas:
         for hdsize in hidden_sizes:
            # Entrenamiento
            print(f"Learning rate {lr} | Lambda {lmbd} | Hidden Size: {hdsize}")
            theta1, theta2, train_losses, val_losses, acc = utils.train(
               X_train_scaled, y_train_np, X_val_scaled, y_val_np, hidden_size= hdsize,
               learning_rate = lr, lambda_= 0.01, epochs= 5000
            )
            print(f"->Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Acc: {acc:.2f}%")

            # Guardamos los datos de cada entrenamiento
            thetas_1.append(theta1)
            thetas_2.append(theta2)
            train_loss_v.append(train_losses)
            val_loss_v.append(val_losses)
            accuracies.append(acc)
            params.append((lr,lmbd,hdsize))

   best_comb = accuracies.index(max(accuracies))
   best_lr, best_lambda, best_hdsize = params[best_comb]
   print(f"\nNº of total evaluated models: {len(accuracies)}")
   print(f"Best parmeters combination:")
   print(f"Learning rate {best_lr} | Lambda {best_lambda} | Hidden Size: {best_hdsize} ")
   print(f"Validation accuracy: {accuracies[best_comb]: .2f}% accuracy")

   #Curvas de aprendizaje
   utils.plot_learning_curves(train_loss_v[best_comb], val_loss_v[best_comb])

   # Modelo con hiperparámetros preestablecidos
   # theta1, theta2, train_losses, val_losses, acc = utils.train(
   #    X_train_scaled, y_train_np, X_val_scaled, y_val_np, hidden_size= 25,
   #    learning_rate = 0.5, lambda_= 1, epochs= 5000
   # )

   # SVM con kernel gaussiano
   # Generamos cada modelo iterando sobre los parámetros posibles
   from sklearn.svm import SVC
   from sklearn.decomposition import PCA
   models_acc = []
   values = [0.01,0.03,0.1,0.3,1,3,10,30]
   y_train_classes = np.argmax(y_train_np, axis=1)
   y_val_classes = np.argmax(y_val_np, axis=1)
   print("\nSVM training:\n ")
   for C in values:
      for sigma in values:
         gamma = 1.0/(2.0*(sigma**2)) 
         svm = SVC(C=C,kernel='rbf', gamma=gamma)
         svm.fit(X_train_scaled,y_train_classes)
         models_acc.append((C,sigma,svm.score(X_val_scaled,y_val_classes)))

   # Ordenamos los modelos por precisión en orden descendente
   ord_models_acc = sorted(models_acc,key=lambda tup: -tup[2])

   # Seleccionamos el modelo óptimo/más preciso
   best_C, best_sigma, best_acc = ord_models_acc[0]
   gamma = 1.0/(2.0*(best_sigma**2)) 
   best_svm = SVC(C=best_C,kernel='rbf', gamma=gamma)
   best_svm.fit(X_train_scaled,y_train_classes)

   # Reducimos dimensión a 2D para visualización
   pca = PCA(n_components=2)
   X_train_2d = pca.fit_transform(X_train_scaled)
   X_val_2d   = pca.transform(X_val_scaled)
   
   # Reentrenamos el SVM sobre las 2 componentes principales
   svm_2d = SVC(C=best_C, kernel='rbf', gamma=gamma)
   svm_2d.fit(X_train_2d, y_train_classes)

   # Mostramos los resultados del modelo y parámetros elegidos para train-set y val-set
   fig, axes1 = plt.subplots(1,1, figsize=(8, 6))
   utils.plot_decision_boundary(svm_2d,X_train_2d,y_train_classes,axes1,best_sigma,sv_train=True)
   fig, axes2 = plt.subplots(1,1, figsize=(8, 6))
   utils.plot_decision_boundary(svm_2d,X_val_2d,y_val_classes,axes2,best_sigma)

   print(f"Nº of total evaluated models: {len(models_acc)}")
   print(f"Best parameters combination: C={best_C} Sigma={best_sigma}")
   print(f"Validation accuracy: {best_acc*100: .2f}%")


   #####################################################
   # 3) EVALUACIÓN

   # Realizamos las predicciones de los distintos modelos sobre el test set
   from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
   X_test_scaled = X_test_scaled.to_numpy()
   y_test_np = pd.get_dummies(y_test).astype(float).to_numpy()
   X_test_2d = pca.transform(X_test_scaled)
   y_test_classes = np.argmax(y_test_np, axis=1)

   print("\nEvaluation of the models:")

   # Red Neuronal
   y_pred_NN = mc.predict(thetas_1[best_comb],thetas_2[best_comb], X_test_scaled)
   acc_NN = np.mean(y_pred_NN == y_test_classes)
   
   # Predicción usando el modelo con parámetros preestablecidos
   # y_pred_NN = mc.predict(theta1,theta2,X_test_scaled)
   # acc_NN = np.mean(y_pred_NN == y_test_classes)

   # SVM
   y_pred_SVM = best_svm.predict(X_test_scaled)
   acc_SVM = accuracy_score(y_test_classes,y_pred_SVM)
   fig, axes3 = plt.subplots(1,1, figsize=(8, 6))
   utils.plot_decision_boundary(svm_2d,X_test_2d,y_test_classes,axes3,best_sigma)

   print(f"Test accuracy Neural Network: {acc_NN*100: .2f}%")
   print(f"Test accuracy SVM: {acc_SVM*100: .2f}%")

   # Matrices de confusión:
   cm_NN = confusion_matrix(y_test_classes,y_pred_NN)
   ConfusionMatrixDisplay(cm_NN, display_labels=target_map).plot(cmap="Blues")
   plt.title("Confusion Matrix - Neural Network")
   cm_SVM = confusion_matrix(y_test_classes,y_pred_SVM)
   ConfusionMatrixDisplay(cm_SVM, display_labels=target_map).plot(cmap="Purples")
   plt.title("Confusion Matrix - SVM RBF")
   plt.show()

   # Análisis de PCA
   pca = PCA()
   pca.fit_transform(X_train_scaled)
   eigenvals = pca.explained_variance_
   iev = pca.explained_variance_ratio_
   cev = np.cumsum(iev)
   n_components_95 = np.argmax(cev >= 0.95) + 1
   n_components_90 = np.argmax(cev >= 0.90) + 1
   print(f"Components for 95% total variance: {n_components_95}")
   print(f"Components for 90% total variance: {n_components_90}")
   utils.plot_pca_analysis(eigenvals,iev,cev)

   n_components = 8  # las que superan Kaiser

   # Varianza explicada ponderada por cada componente
   print("\nGlobal contribution for main 8 Principal Components")
   importance = pd.Series(
      np.sum(np.abs(pca.components_[:n_components]), axis=0),
      index= X.columns.to_list()
   ).sort_values(ascending=False)
   print(importance)

   # Visualizar
   importance.plot(kind="bar", figsize=(12, 4), title="Contribución global de features (primeras 8 PCs)")
   plt.xticks(rotation=45, ha="right")
   plt.tight_layout()
   plt.show()

   # NOTA: Vemos que la edad, "age", es una de las características que más valor aporta según el PCA, lo cual resulta
   # algo llamativo, pues todos los alumnos están en el mismo rango de edad [17,24] con una distribución muy
   # similar en cada edad. Lo podes ver así:

   # Y además vemos que la correlación de "age" con cualqiera del resto de componentes es inferior a 0.05
   # Probamos a eliminarla
   #correlaciones_age = X.corr()["age"].abs().sort_values(ascending=False)
   #print(correlaciones_age)
   
   



if __name__ == "__main__":
   main()