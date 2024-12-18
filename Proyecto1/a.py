import pygame
from queue import PriorityQueue
from typing import List

pygame.init()

icono = pygame.image.load("C:/Users/morel/Downloads/Iaaa/Ia/Proyecto1/icono.jpg")
pygame.display.set_icon(icono)

heigthventana = 600
window = pygame.display.set_mode((heigthventana, heigthventana))
pygame.display.set_caption("paseme profe")
font = pygame.font.SysFont(None, 78)


blanquito = (240, 240, 240)
Rojito = (211, 25, 56)
gris = (200, 200, 200)
success = (0, 255, 4)
rojito = (255, 0, 0 )
inicio = (0, 50, 255 )
fin = (182, 255, 0)
trazo = (0, 255, 220 )

costDiag = 1.4
costRect = 1.0


class Nodo:
    def __init__(self, fila, columna, ancho, total_filas):
        self.fila = fila
        self.columna = columna
        self.x = fila * ancho
        self.y = columna * ancho
        self.color = blanquito
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []
        self.padre = None

    def get_pos(self):
        return self.fila, self.columna

    def chocaste_paps(self):
        return self.color == Rojito

    def empiezelamasacreprofe(self):
        return self.color == inicio

    def ayuda(self):
        return self.color == fin

    def vasallorar(self):
        self.color = blanquito

    def arrancamos(self):
        self.color = inicio

    def hacer_pared(self):
        self.color = Rojito

    def hacer_fin(self):
        self.color = fin

    def nopuedespasarpaps(self):
        self.color = rojito

    def sipuedespasarpaps(self):
        self.color = success

    def paselealobarrido(self):
        self.color = trazo

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_noditos(self, grid: List[List["Nodo"]], filas: int):
        self.vecinos = []
       
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.columna].chocaste_paps():
            self.vecinos.append((grid[self.fila + 1][self.columna], costRect))
   
        if self.fila > 0 and not grid[self.fila - 1][self.columna].chocaste_paps():
            self.vecinos.append((grid[self.fila - 1][self.columna], costRect))
        if self.columna < self.total_filas - 1 and not grid[self.fila][self.columna + 1].chocaste_paps():
            self.vecinos.append((grid[self.fila][self.columna + 1], costRect))
        if self.columna > 0 and not grid[self.fila][self.columna - 1].chocaste_paps():
            self.vecinos.append((grid[self.fila][self.columna - 1], costRect))
            
        if (self.fila < self.total_filas - 1 and self.columna < self.total_filas - 1 and
            not grid[self.fila + 1][self.columna + 1].chocaste_paps() and
            (not grid[self.fila + 1][self.columna].chocaste_paps() or not grid[self.fila][self.columna + 1].chocaste_paps())):
            self.vecinos.append((grid[self.fila + 1][self.columna + 1], costDiag))

        if (self.fila < self.total_filas - 1 and self.columna > 0 and
            not grid[self.fila + 1][self.columna - 1].chocaste_paps() and
            (not grid[self.fila + 1][self.columna].chocaste_paps() or not grid[self.fila][self.columna - 1].chocaste_paps())):
            self.vecinos.append((grid[self.fila + 1][self.columna - 1], costDiag))

        if (self.fila > 0 and self.columna < self.total_filas - 1 and
            not grid[self.fila - 1][self.columna + 1].chocaste_paps() and
            (not grid[self.fila - 1][self.columna].chocaste_paps() or not grid[self.fila][self.columna + 1].chocaste_paps())):
            self.vecinos.append((grid[self.fila - 1][self.columna + 1], costDiag))

        if (self.fila > 0 and self.columna > 0 and
            not grid[self.fila - 1][self.columna - 1].chocaste_paps() and
            (not grid[self.fila - 1][self.columna].chocaste_paps() or not grid[self.fila][self.columna - 1].chocaste_paps())):
            self.vecinos.append((grid[self.fila - 1][self.columna - 1], costDiag))



def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid


def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, gris, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, gris, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))


def dibujar(ventana, grid, filas, ancho):
    ventana.fill(blanquito)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()


def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    columna = x // ancho_nodo
    return fila, columna


def heuristica(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        actual.paselealobarrido()
        dibujar()
        pygame.time.delay(100)


def algoritmo_a(dibujar, grid, inicio, fin):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, inicio))
    came_from = {}
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}

    while not open_set.empty():
        pygame.time.delay(50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        actual = open_set.get()[2]
        open_set_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            inicio.arrancamos()
            return True

        for vecino, costo in actual.vecinos:
            temp_g_score = g_score[actual] + costo

            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + heuristica(vecino.get_pos(), fin.get_pos())
                if vecino not in open_set_hash:
                    count += 1
                    open_set.put((f_score[vecino], count, vecino))
                    open_set_hash.add(vecino)
                    vecino.sipuedespasarpaps()

        dibujar()

        if actual != inicio:
            actual.nopuedespasarpaps()

    return False


def main(ventana, ancho):
    while True:
        try:
            FILAS = int(input("Ingrese el tamaño de la matriz (por ejemplo, 10): "))
            if FILAS > 0:
                break
        except ValueError:
            print("Por favor, ingrese un número válido.")

    grid = crear_grid(FILAS, ancho)
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, columna = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][columna]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.arrancamos()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]: 
                pos = pygame.mouse.get_pos()
                fila, columna = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][columna]
                nodo.vasallorar()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_noditos(grid, FILAS)

                    algoritmo_a(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

    pygame.quit()


main(window, heigthventana)
