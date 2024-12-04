
import numpy as np
import matplotlib.pyplot as plt


X_prod = np.array([[0.5, 0.8], [0.6, 0.9], [0.7, 0.6], [0.4, 0.5], [0.3, 0.9], [0.8, 0.4]])
y_prod = np.array([1, 1, 0, 0, 1, 0])

weights = np.random.rand(2)  
bias = np.random.rand(1)     
learning_rate = 0.1         
epochs = 100                

for epoch in range(epochs):
    for i in range(len(X_prod)):
        linear_output = np.dot(X_prod[i], weights) + bias
        prediction = 1 if linear_output >= 0 else 0
        error = y_prod[i] - prediction
        weights += learning_rate * error * X_prod[i]
        bias += learning_rate * error

print("Pesos finales (problema 1):", weights)
print("Sesgo final (problema 1):", bias)

x_vals = np.linspace(0, 1, 100)
y_vals = -(weights[0] * x_vals + bias) / weights[1]

plt.figure(figsize=(8, 6))
plt.scatter(X_prod[:, 0], X_prod[:, 1], c=y_prod, cmap='viridis')
plt.plot(x_vals, y_vals, color='red', label='Frontera de decisión')
plt.xlabel('Precio relativo')
plt.ylabel('Calidad percibida')
plt.legend()
plt.title('Frontera de decisión (Problema 1)')
plt.show()

