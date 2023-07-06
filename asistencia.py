import cv2
import face_recognition as fr
import os

# crear base de datos
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}\{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

# Codificar imagenes
def codificar(imagenes):

    # crear una lista nueva
    lista_codificada = []

    # pasar todas las imagenes a rgb
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Codificar
        codificado = fr.face_encodings(imagen)[0]

        # Agregar a la lista
        lista_codificada.append(codificado)

    # devolver lista codificada
    return lista_codificada

lista_empleados_codificada = codificar(mis_imagenes)
print(len(lista_empleados_codificada))