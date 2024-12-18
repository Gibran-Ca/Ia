import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D
)
from keras.layers import LeakyReLU
from skimage.transform import resize

imgpath = 'C:/Users/morel/Downloads/dataset'  

images = []
labels = []
directories = []
dircount = []

image_size = (128, 128)  #

print("Leyendo imágenes de", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if len(image.shape) == 3:  
                image_resized = resize(image, image_size, anti_aliasing=True, clip=False, preserve_range=True)
                images.append(image_resized)
                label = root.split(os.sep)[-1]  
                labels.append(label)

class_names = sorted(list(set(labels))) 
nClasses = len(class_names)
class_map = {class_name: index for index, class_name in enumerate(class_names)} 
print('Clases encontradas:', class_names)

y = np.array([class_map[label] for label in labels])

train_X, test_X, train_Y, test_Y = train_test_split(images, y, test_size=0.2)
print('Tamaño de los datos de entrenamiento:', train_X.shape, train_Y.shape)
print('Tamaño de los datos de prueba:', test_X.shape, test_Y.shape)

train_X = np.array(train_X).astype('float32') / 255.
test_X = np.array(test_X).astype('float32') / 255.

train_Y_one_hot = to_categorical(train_Y, nClasses)
test_Y_one_hot = to_categorical(test_Y, nClasses)

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

INIT_LR = 1e-3 
epochs = 35 
batch_size = 64 

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(train_X.shape[1], train_X.shape[2], 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))  

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, decay=INIT_LR / 100), metrics=['accuracy'])

history = model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(accuracy))

plt.plot(epochs_range, accuracy, 'bo', label='Precisión de entrenamiento')
plt.plot(epochs_range, val_accuracy, 'b', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Pérdida de entrenamiento')
plt.plot(epochs_range, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.legend()
plt.show()

# Guardar el modelo entrenado en un archivo .h5
model.save('modeloAutos.h5')
print("Modelo guardado en 'modelo_entrenado.h5'")

predicted_classes = model.predict(test_X)

predicted_classes = np.argmax(predicted_classes, axis=1)

target_names = class_names
print(classification_report(test_Y, predicted_classes, target_names=target_names))

correct = np.where(predicted_classes == test_Y)[0]
print(f"Se encontraron {len(correct)} etiquetas correctas")

for i, correct_idx in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[correct_idx])
    plt.title(f"{class_names[predicted_classes[correct_idx]]}, {class_names[test_Y[correct_idx]]}")
    plt.tight_layout()

# Imágenes incorrectas
incorrect = np.where(predicted_classes != test_Y)[0]
print(f"Se encontraron {len(incorrect)} etiquetas incorrectas")

for i, incorrect_idx in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[incorrect_idx])
    plt.title(f"{class_names[predicted_classes[incorrect_idx]]}, {class_names[test_Y[incorrect_idx]]}")
    plt.tight_layout()

filenames = ['C:/Users/morel/Downloads/pygamescc/proyecto3/prueba1.jpg']  

images = []
for filepath in filenames:
    image = plt.imread(filepath)
    image_resized = resize(image, (128, 128), anti_aliasing=True, clip=False, preserve_range=True)  # Redimensionar
    images.append(image_resized)

X_new = np.array(images, dtype=np.uint8)
X_new = X_new.astype('float32') / 255.

predicted_classes_new = model.predict(X_new)

for i, img_tagged in enumerate(predicted_classes_new):
    print(f"Imagen: {filenames[i]} - Predicción: {class_names[np.argmax(img_tagged)]}")
