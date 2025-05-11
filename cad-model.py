import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random
import pathlib
import cv2
import pandas

# Definir el directorio de datos
data_dir = pathlib.Path("datasets/cats_and_dogs_filtered/train")

# Listar las carpetas (deberían ser ['cats', 'dogs'])
folders = os.listdir(data_dir)
print("Carpetas encontradas:", folders)

# Inicializar listas para las imágenes y etiquetas
image_names = []
train_labels = []
train_images = []

# Definir el tamaño de las imágenes
size = 64, 64

# Cargar las imágenes y las etiquetas
for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            img_path = os.path.join(folder_path, file)
            image_names.append(img_path)
            train_labels.append(folder)  # 'cats' o 'dogs'
            img = cv2.imread(img_path)
            if img is not None:
                im = cv2.resize(img, size)
                train_images.append(im)

# Convertir las imágenes a un array numpy
train = np.array(train_images)
print("Forma del conjunto de entrenamiento:", train.shape)

# Normalizar las imágenes
train = train.astype('float32') / 255.0

# Convertir etiquetas a valores numéricos
label_dummies = pandas.get_dummies(train_labels)
labels = label_dummies.values.argmax(1)

# Barajar datos
union_list = list(zip(train, labels))
random.shuffle(union_list)
train, labels = zip(*union_list)

# Convertir a numpy
train = np.array(train)
labels = np.array(labels)

# Crear modelo secuencial
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64, 3)),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(2, activation='softmax')  # Dos clases: gatos y perros
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train, labels, epochs=7)

# Guardar el modelo en formato .keras
model.save('modelo_cad.keras')

print("Modelo guardado en formato .keras.")
