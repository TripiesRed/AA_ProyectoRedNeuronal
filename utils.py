import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import logistic_reg as lgr
import multi_class as mc

def cost(theta1, theta2, X, y, lambda_):
   """
   Compute cost for 2-layer neural network. 

   Parameters
   ----------
   theta1 : array_like
      Weights for the first layer in the neural network.
      It has shape (2nd hidden layer size x input size + 1)

   theta2: array_like
      Weights for the second layer in the neural network. 
      It has shape (output layer size x 2nd hidden layer size + 1)

   X : array_like
      The inputs having shape (number of examples x number of dimensions).

   y : array_like
      1-hot encoding of labels for the input, having shape 
      (number of examples x number of labels).

   lambda_ : float
      The regularization parameter. 

   Returns
   -------
   J : float
      The computed value for the cost function. 

   """

   m = X.shape[0]

   A1, A2, H = mc.forward_propagation(theta1, theta2, X)
   H = np.clip(H, 1e-15, 1 - 1e-15) #####
   J = np.sum(-y*np.log(H) - (1-y)*np.log(1-H))/m
   J += (lambda_/(2*m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))

   return J


def backprop(theta1, theta2, X, y, lambda_):
   """
   Compute cost and gradient for 2-layer neural network. 

   Parameters
   ----------
   theta1 : array_like
      Weights for the first layer in the neural network.
      It has shape (2nd hidden layer size x input size + 1)

   theta2: array_like
      Weights for the second layer in the neural network. 
      It has shape (output layer size x 2nd hidden layer size + 1)

   X : array_like
      The inputs having shape (number of examples x number of dimensions).

   y : array_like
      1-hot encoding of labels for the input, having shape 
      (number of examples x number of labels).

   lambda_ : float
      The regularization parameter. 

   Returns
   -------
   J : float
      The computed value for the cost function. 

   grad1 : array_like
      Gradient of the cost function with respect to weights
      for the first layer in the neural network, theta1.
      It has shape (2nd hidden layer size x input size + 1)

   grad2 : array_like
      Gradient of the cost function with respect to weights
      for the second layer in the neural network, theta2.
      It has shape (output layer size x 2nd hidden layer size + 1)

   """
   m = X.shape[0]
   A1, A2, H = mc.forward_propagation(theta1, theta2, X)

   Delta1 = np.zeros(theta1.shape)
   Delta2 = np.zeros(theta2.shape)
   H = np.clip(H, 1e-15, 1 - 1e-15) ####
   J = cost(theta1, theta2, X, y, lambda_)
   D3 = H - y
   D2 = np.dot(theta2.T, D3.T).T * (A2*(1-A2))

   Delta1 += np.dot((D2[:, 1:]).T, A1)
   Delta2 += np.dot(D3.T, A2)

   grad1 = Delta1 / m
   grad2 = Delta2 / m
   
   grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
   grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]

   return (J, grad1, grad2)


def train(X_train, y_train, X_val, y_val, hidden_size=25, learning_rate=0.1,
          lambda_=0.01, epochs=1000):

   # Inicialización aleatoria de pesos 
   input_size  = X_train.shape[1]
   output_size = y_train.shape[1]
   epsilon = 0.12

   theta1 = np.random.uniform(-epsilon, epsilon, (hidden_size, input_size + 1))
   theta2 = np.random.uniform(-epsilon, epsilon, (output_size, hidden_size + 1))

   train_losses = []
   val_losses   = []

   for epoch in range(epochs):

      # Gradientes via backprop
      train_loss, grad1, grad2 = backprop(theta1, theta2, X_train, y_train, lambda_)

      # Actualización de parámetros
      theta1 -= learning_rate * grad1
      theta2 -= learning_rate * grad2

      # Registro de pérdidas
      val_loss = cost(theta1, theta2, X_val,   y_val,   lambda_)
      train_losses.append(train_loss)
      val_losses.append(val_loss)

      if epoch % 100 == 0:
         preds = mc.predict(theta1, theta2, X_train)
         y_train_classes = np.argmax(y_train, axis=1)
         acc = np.mean(preds == y_train_classes) * 100
         print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.1f}%")

   return theta1, theta2, train_losses, val_losses, acc

def plot_learning_curves(train_losses, val_losses):

   plt.figure(figsize=(8, 4))
   plt.plot(train_losses, label="Train loss")
   plt.plot(val_losses, label="Validation loss")
   plt.xlabel("Epoch")
   plt.ylabel("Cost")
   plt.title("Learning curves")
   plt.legend()
   plt.tight_layout()
   plt.show()