import io
import json
import numpy as np
from PIL import Image
import requests
from numpy import asarray

# URL del servidor con el modelo desplegado
SERVER_URL = 'https://tensorflow-cad.onrender.com/v1/models/cad_modelo:predict'

def main():
    # Cargar la imagen desde el archivo local
    img = Image.open('dog.2.jpg')  # Cambia el nombre por el de tu imagen
    img = img.resize((64, 64))  # Redimensionar la imagen al tamaño de entrada de tu modelo
    img_array = asarray(img)  # Convertir la imagen a un array numpy

    # Expandir dimensiones para que el array tenga la forma adecuada para el modelo
    img_array = np.expand_dims(img_array, 0).tolist()  # Convertir a lista

    # Crear el cuerpo de la solicitud con la imagen
    predict_request = json.dumps({'instances': img_array})

    # Enviar la solicitud de predicción al servidor
    response = requests.post(SERVER_URL, data=predict_request)

    # Manejo de errores en caso de una respuesta incorrecta
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Error en la solicitud: {err}")
        return

    # Obtener la predicción del modelo
    prediction = response.json()['predictions'][0]
    print("Predicción del modelo:", prediction)

    # Etiquetas de clases (ajusta según las clases de tu modelo)
    classes_labels = ['gato', 'perro']  # Cambia las etiquetas según tu modelo
    predicted_class = classes_labels[np.argmax(prediction)]  # Obtener la clase predicha

    print(f"La clase predicha es: {predicted_class}")
    print(f"Índice de la clase predicha: {np.argmax(prediction)}")


if __name__ == '__main__':
    main()
