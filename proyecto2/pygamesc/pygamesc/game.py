import pygame
import random
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from red_neuronal import entrenar_red_neuronal
from arbol_decision import entrenar_arbol_decision
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('pygamesc/assets/sprites/mono_frame_1.png'),
    pygame.image.load('pygamesc/assets/sprites/mono_frame_2.png'),
    pygame.image.load('pygamesc/assets/sprites/mono_frame_3.png'),
    pygame.image.load('pygamesc/assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('pygamesc/assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('pygamesc/assets/game/fondo2.png')
nave_img = pygame.image.load('pygamesc/assets/game/ufo.png')
menu_img = pygame.image.load('pygamesc/assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# Cargar el modelo entrenado de la red neuronal
#modelo_red_neuronal = None
#modelo = load_model("modelo_red_neuronal.h5")

#modelo_red_neuronal = load_model("modelo_red_neuronal.h5")

def cargar_modelo():
    global modelo_red_neuronal
    try:
        modelo_red_neuronal = load_model("modelo_red_neuronal.h5")  # Cargar modelo .h5
        print("Modelo cargado correctamente.")
    except FileNotFoundError:
        print("No se encontró un modelo previamente entrenado.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

# Función para predecir el salto usando el modelo de red neuronal
def predecir_salto(velocidad_bala, distancia):
    if modelo_red_neuronal:
        # Convertir la entrada a un array NumPy compatible con el modelo
        entrada = np.array([[velocidad_bala, distancia]], dtype=np.float32)
        
        # Realizar la predicción
        prediccion = modelo_red_neuronal.predict(entrada)
        
        # Mostrar la predicción
        print("Valor de predicción:", prediccion[0][0])
        
        # Determinar si debe saltar según el umbral (0.5)
        if prediccion[0][0] >= 0.5:
            return 1  # Saltar
        else:
            return 0  # No saltar
    
    # Si no hay modelo cargado, no saltar
    print("No se ha cargado un modelo.")
    return 0

# Función para cargar el modelo entrenado de Árbol de Decisión
def cargar_modelo_arbol():
    global modelo_arbol_decision
    try:
        with open("modelo.pkl", "rb") as archivo_modelo:
            modelo_arbol_decision = pickle.load(archivo_modelo)
            print("Modelo de Árbol de Decisión cargado correctamente.")
    except FileNotFoundError:
        print("No se encontró un modelo previamente entrenado de Árbol de Decisión.")

# Función para predecir el salto usando el modelo de Árbol de Decisión
def predecir_salto_arbol(velocidad_bala, distancia):
    if modelo_arbol_decision:
        prediccion = modelo_arbol_decision.predict([[velocidad_bala, distancia]])  # Usamos la predicción del modelo
        #print("Valor Predicho por Árbol de Decisión:", prediccion[0])
        return prediccion[0]  # Retorna 1 si saltó, 0 si no saltó
    return 0  # Si no hay modelo cargado, no saltar

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú

# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0  # 1 si saltó, 0 si no saltó
    # Guardar velocidad de la bala, distancia al jugador y si saltó o no
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))
    
def graficar_datos():
    # Separar datos según el valor de 'salto_hecho'
    x1 = [x for x, y, z in datos_modelo if z == 0]
    x2 = [y for x, y, z in datos_modelo if z == 0]
    target0 = [z for x, y, z in datos_modelo if z == 0]

    x3 = [x for x, y, z in datos_modelo if z == 1]
    x4 = [y for x, y, z in datos_modelo if z == 1]
    target1 = [z for x, y, z in datos_modelo if z == 1]

    # Crear el gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos con salto_hecho=0 en azul
    ax.scatter(x1, x2, target0, c='blue', marker='o', label='Target=0')

    # Graficar los puntos con salto_hecho=1 en rojo
    ax.scatter(x3, x4, target1, c='red', marker='x', label='Target=1')

    # Etiquetas y leyenda
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('target')
    ax.legend()

    plt.show()
    
    
def graficar_arbol():
    # Separar datos
    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    # Definir características (X) y etiquetas (y)
    X = list(zip(x1, x2))  # Las dos primeras columnas son las características
    y = target0  # La tercera columna es la etiqueta

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de Árbol de Decisión
    clf = DecisionTreeClassifier()

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Exportar el árbol de decisión en formato DOT para su visualización
    dot_data = export_graphviz(clf, out_file=None, 
                            feature_names=['Feature 1', 'Feature 2'],  
                            class_names=['Clase 0', 'Clase 1'],  
                            filled=True, rounded=True,  
                            special_characters=True)  

    # Crear el gráfico con graphviz
    graph = graphviz.Source(dot_data)

    # Mostrar el gráfico
    graph.view()

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")


# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto
    global modelo_actual 
    pantalla.fill(NEGRO)
    texto = fuente.render("1.- Juego Manual", True, BLANCO)
    texto1 = fuente.render("2.- Redes Neuronales", True, BLANCO)
    texto2 = fuente.render("3.- Árbol de Decisión", True, BLANCO)
    texto3 = fuente.render("4.- Salir", True, BLANCO)
    x_centro = w // 2
    y_inicial = h // 3
    espacio_entre_renglones = 50  # Espaciado entre cada renglón

    pantalla.blit(texto, (x_centro - texto.get_width() // 2, y_inicial))
    pantalla.blit(texto1, (x_centro - texto1.get_width() // 2, y_inicial + espacio_entre_renglones))
    pantalla.blit(texto2, (x_centro - texto2.get_width() // 2, y_inicial + 2 * espacio_entre_renglones))
    pantalla.blit(texto3, (x_centro - texto3.get_width() // 2, y_inicial + 3 * espacio_entre_renglones))
    pygame.display.flip() 

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:   
                if evento.key == pygame.K_1:
                    print("MODO MANUAL")
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_2:
                    print("REDES NEURONALES")
                    modo_auto = True
                    menu_activo = False
                    entrenar_red_neuronal(datos_modelo)
                    cargar_modelo()
                    graficar_datos()
                    modelo_actual = "red_neuronal"
                elif evento.key == pygame.K_3:
                    print("ARBOL DE DECISION")
                    modo_auto = True
                    menu_activo = False
                    entrenar_arbol_decision(datos_modelo)
                    cargar_modelo_arbol()
                    graficar_arbol()
                    modelo_actual = "arbol"
                elif evento.key == pygame.K_4:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    en_suelo = True
    # Mostrar los datos recopilados hasta el momento
    print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo

def main():
    global salto, en_suelo, bala_disparada

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual: el jugador controla el salto
            if not modo_auto:
                if salto:
                    manejar_salto()
                # Guardar los datos si estamos en modo manual
                guardar_datos()
            else:
                distancia = abs(jugador.x - bala.x)
                # Modo automático: el modelo decide si saltar
                if modelo_actual == "arbol":
                    # Usamos el modelo de árbol de decisión
                    decision_salto = predecir_salto_arbol(velocidad_bala, distancia)
                    #print("Arbol: ", decision_salto)
                elif modelo_actual == "red_neuronal":
                    # Usamos el modelo de redes neuronales
                    decision_salto = predecir_salto(velocidad_bala, distancia)
                   # print("REdes neu: ", decision_salto)
                else:
                    #No se selecciono ningun modelo
                    decision_salto = 0

                if decision_salto == 1 and en_suelo:
                    salto = True
                    en_suelo = False
                if salto:
                    manejar_salto()
            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
