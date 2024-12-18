import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

modelo_h5 = 'C:/Users/morel/Downloads/pygamescc/proyecto3v2/modeloAutos.h5'  
image_path = 'C:/Users/morel/Downloads/pygamescc/proyecto3v2/463091357_856576499991180_7708548709456031439_n.jpg'  
sriesgos = ['ibiza', 'camaro', 'clasico', 'mk5', 'cx5'] 

IMG_SIZE = (224, 224)  

model = load_model(modelo_h5)
print("Modelo cargado con éxito.")

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  
    img_array = img_to_array(img) / 255.0 
    return np.expand_dims(img_array, axis=0) 

input_image = preprocess_image(image_path, IMG_SIZE)

predictions = model.predict(input_image)
predicted_class_index = np.argmax(predictions[0])  
predicted_label = sriesgos[predicted_class_index] 


print(f"La imagen evaluada pertenece a la clase: {predicted_label}")
print(f"Confianza de la predicción: {predictions[0][predicted_class_index] * 100:.2f}%")

plt.imshow(load_img(image_path))
plt.title(f"Predicción: {predicted_label}")
plt.axis('off')
plt.show()
