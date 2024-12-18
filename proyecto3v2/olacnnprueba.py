import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tensorflow as tf

IMG_SIZE = (224, 224)  
DATASET_PATH = "C:/Users/morel/Downloads/dataset"  

model = load_model("C:/Users/morel/Downloads/pygamescc/proyecto3v2/modeloAutos.h5")
print("Modelo cargado con éxito.")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
obtener_clase = datagen.flow_from_directory(
    DATASET_PATH,
    class_mode='categorical'
)











def predict_image(image_path, model, class_indices):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  

    class_names = list(class_indices.keys())
    return class_names[predicted_class]

image_path = "C:/Users/morel/Downloads/pygamescc/proyecto3v2/469358474_595722829710917_2749896076155422514_n.jpg" 
# image_path1 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba2.jpg" 
# image_path2 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba3.jpg" 
# image_path3 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba4.jpg" 
# image_path4 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba5.jpg" 
# image_path5 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba6.jpg" 
# image_path6 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba7.jpg" 
# image_path7 = "C:/Users/morel/Downloads/pygamescc/proyecto3/prueba8.jpg" 




predicted_label = predict_image(image_path, model, obtener_clase.class_indices)
# predicted_label1 = predict_image(image_path1, model, obtener_clase.class_indices)
# predicted_label2 = predict_image(image_path2, model, obtener_clase.class_indices)
# predicted_label3 = predict_image(image_path3, model, obtener_clase.class_indices)
# predicted_label4 = predict_image(image_path4, model, obtener_clase.class_indices)
# predicted_label5 = predict_image(image_path5, model, obtener_clase.class_indices)
# predicted_label6 = predict_image(image_path6, model, obtener_clase.class_indices)
# predicted_label7 = predict_image(image_path7, model, obtener_clase.class_indices)

print("El modelo predice que la imagen es:", predicted_label)
# print("El modelo predice que la imagen es:", predicted_label1)
# print("El modelo predice que la imagen es:", predicted_label2)
# print("El modelo predice que la imagen es:", predicted_label3)
# print("El modelo predice que la imagen es:", predicted_label4)
# print("El modelo predice que la imagen es:", predicted_label5)
# print("El modelo predice que la imagen es:", predicted_label6)
# print("El modelo predice que la imagen es:", predicted_label7)


