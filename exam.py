import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_clientes = np.array([[0.9, 0.8, 0.2], [0.7, 0.6, 0.5], [0.4, 0.4, 0.8],
                       [0.8, 0.9, 0.3], [0.5, 0.7, 0.6], [0.3, 0.5, 0.9]])
y_clientes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

model = Sequential([
    Dense(10, activation='relu', input_shape=(3,)),  
    Dense(3, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_clientes, y_clientes, epochs=500, verbose=1)

predictions = model.predict(X_clientes)

weights_hidden = model.layers[0].get_weights()  # Pesos de la capa oculta
weights_output = model.layers[1].get_weights()  # Pesos de la capa de salida

output_text = """
Proceso y resultados del modelo de red neuronal

 Creacion del modelo de red neuronal

Se ha creado un modelo secuencial con dos capas
Primera capa oculta 10 neuronas con funcion de activacion ReLU
Segunda capa de salida 3 neuronas con funcion de activación softmax


 Resultados del entrenamiento



Predicciones Problema 2
""" + "\n".join([str(pred) for pred in predictions]) + "\n\n"

output_text += "Pesos de la primera capa oculta 10 neuronas\n" + str(weights_hidden) + "\n\n"
output_text += "Pesos de la segunda capa de salida 3 neuronas\n" + str(weights_output)

file_path = 'resultados_examen_CAPJG.txt'

with open(file_path, 'w') as file:
    file.write(output_text)

print(f"Archivo guardado en: {file_path}")
