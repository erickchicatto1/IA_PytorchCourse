"""
Dijkstra con Pygame - Navegacion de grafos
==========================================
Instalacion:
  pip install pygame

Uso:
  python dijkstra_pygame.py

Controles por CONSOLA:
  inicio <NODO>   → ej: inicio A
  meta <NODO>     → ej: meta O
  empezar         → ejecuta Dijkstra
  reiniciar       → limpia todo
  nodos           → lista nodos disponibles
  ayuda           → muestra comandos
  salir           → cierra el programa
  
  
  https://www.datacamp.com/tutorial/dijkstra-algorithm-in-python
  
"""

import pygame
import time
import threading
import sys
from heapq import heapify, heappop, heappush

# ══════════════════════════════════════════════════════════════════════════════
#  CLASE GRAPH  (implementacion de Dijkstra)
# ══════════════════════════════════════════════════════════════════════════════

class Graph:
    def __init__(self, graph: dict = {}):
        self.graph = graph

    def shortest_distances(self, source: str):
        """Devuelve (distances, parents) desde el nodo source."""
        # Inicializar distancias con infinito
        distances = {node: float("inf") for node in self.graph}
        distances[source] = 0

        # Cola de prioridad
        pq = [(0, source)]
        heapify(pq)

        # Nodo padre para reconstruir la ruta
        parents = {node: None for node in self.graph}

        visited = set()

        while pq:
            current_distance, current_node = heappop(pq)

            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parents[neighbor]   = current_node
                    heappush(pq, (distance, neighbor))

        return distances, parents

    def shortest_path(self, source: str, target: str):
        """Devuelve (ruta_como_lista, costo_total)."""
        distances, parents = self.shortest_distances(source)

        if distances[target] == float("inf"):
            return [], float("inf")   # sin ruta

        # Reconstruir camino
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = parents[current]
        path.reverse()

        return path, distances[target]


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACION VISUAL
# ══════════════════════════════════════════════════════════════════════════════

ANCHO, ALTO = 960, 680
FPS         = 60
RADIO       = 24

C_FONDO    = ( 20,  20,  46)
C_ARISTA   = ( 74,  74, 106)
C_NODO     = ( 45,  45,  78)
C_BORDE    = (127, 119, 221)
C_TEXTO    = (224, 224, 240)
C_INICIO   = ( 29, 158, 117)
C_META     = (216,  90,  48)
C_VISITADO = ( 83,  74, 183)
C_RUTA     = (239, 159,  39)
C_PESO     = (159, 225, 203)
C_PANEL    = ( 12,  12,  30)
C_BLANCO   = (255, 255, 255)
C_GRIS     = (150, 150, 150)
C_VERDE    = ( 29, 158, 117)
C_ROJO     = (216,  90,  48)

# ══════════════════════════════════════════════════════════════════════════════
#  DATOS DEL GRAFO
# ══════════════════════════════════════════════════════════════════════════════

# Posiciones en pantalla
NODOS = {
    "A": (120, 150), "B": (260,  80), "C": (400,  80),
    "D": (540,  80), "E": (680, 150), "F": (120, 280),
    "G": (260, 260), "H": (400, 280), "I": (540, 260),
    "J": (680, 280), "K": (120, 420), "L": (260, 450),
    "M": (400, 480), "N": (540, 450), "O": (680, 420),
}

# Lista de aristas: (nodo_a, nodo_b, peso)
ARISTAS = [
    ("A","B", 4), ("A","F", 3), ("B","C", 5), ("B","G", 7),
    ("C","D", 3), ("C","H", 4), ("D","E", 6), ("D","I", 5),
    ("E","J", 2), ("F","G", 6), ("F","K", 5), ("G","H", 3),
    ("G","L", 8), ("H","I", 4), ("H","M", 6), ("I","J", 3),
    ("I","N", 7), ("J","O", 4), ("K","L", 5), ("L","M", 3),
    ("L","N", 9), ("M","N", 4), ("N","O", 5), ("B","D", 9),
    ("F","H", 8), ("K","M", 7),
]

# Construir diccionario de adyacencia para la clase Graph
grafo_dict = {n: [] for n in NODOS}
for u, v, w in ARISTAS:
    grafo_dict[u].append((v, w))
    grafo_dict[v].append((u, w))

# Instancia de la clase Graph
g = Graph(grafo_dict)


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADO COMPARTIDO  (hilo consola <-> hilo principal Pygame)
# ══════════════════════════════════════════════════════════════════════════════

lock        = threading.Lock()
inicio      = None
meta        = None
en_proceso  = False
estados     = {}          # nodo -> color RGB
ruta_linea  = []          # segmentos [(x1,y1,x2,y2), ...]
robot_pos   = None        # (x, y) posicion actual del robot animado
mensaje     = "Escribe en la consola:  inicio <N>   meta <N>   empezar"
corriendo   = True


# ══════════════════════════════════════════════════════════════════════════════
#  DIBUJO  (solo llamado desde el hilo principal de Pygame)
# ══════════════════════════════════════════════════════════════════════════════

def _color_nodo(nombre):
    with lock:
        ini, met, est = inicio, meta, estados.copy()
    if nombre == ini: return C_INICIO
    if nombre == met: return C_META
    return est.get(nombre, C_NODO)

def _segmento_entre_nodos(p1, p2):
    """Devuelve el segmento de linea recortado en los bordes del circulo."""
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    dist   = (dx**2 + dy**2) ** 0.5
    if dist == 0:
        return p1, p2
    ux, uy = dx/dist, dy/dist
    return (
        (int(p1[0] + ux*RADIO), int(p1[1] + uy*RADIO)),
        (int(p2[0] - ux*RADIO), int(p2[1] - uy*RADIO)),
    )

def dibujar_grafo(surf, font_peso, font_nodo):
    # 1. Aristas base
    for u, v, w in ARISTAS:
        p1, p2 = NODOS[u], NODOS[v]
        a, b   = _segmento_entre_nodos(p1, p2)
        pygame.draw.line(surf, C_ARISTA, a, b, 2)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        txt = font_peso.render(str(w), True, C_PESO)
        surf.blit(txt, txt.get_rect(center=(mx, my-11)))

    # 2. Ruta optima resaltada
    with lock:
        rl = ruta_linea.copy()
    for x1, y1, x2, y2 in rl:
        pygame.draw.line(surf, C_RUTA, (x1,y1), (x2,y2), 5)

    # 3. Nodos
    for nombre, (x, y) in NODOS.items():
        fill  = _color_nodo(nombre)
        borde = {C_INICIO: (15,110,86), C_META: (153,60,29),
                 C_RUTA: (133,79,11)}.get(fill, C_BORDE)
        pygame.draw.circle(surf, fill,  (x, y), RADIO)
        pygame.draw.circle(surf, borde, (x, y), RADIO, 2)
        txt = font_nodo.render(nombre, True, C_TEXTO)
        surf.blit(txt, txt.get_rect(center=(x, y)))

    # 4. Robot
    with lock:
        rp = robot_pos
    if rp:
        pygame.draw.circle(surf, C_RUTA,   rp, 11)
        pygame.draw.circle(surf, C_BLANCO, rp, 11, 2)

def dibujar_panel(surf, font_msg, font_small):
    panel_y = ALTO - 90
    pygame.draw.rect(surf, C_PANEL, (0, panel_y, ANCHO, 90))
    pygame.draw.line(surf, C_BORDE, (0, panel_y), (ANCHO, panel_y), 1)

    with lock:
        msg, ini, met = mensaje, inicio, meta

    if ini:
        pygame.draw.circle(surf, C_INICIO, (28, panel_y+22), 8)
        surf.blit(font_small.render(f"Inicio: {ini}", True, C_VERDE), (44, panel_y+14))
    if met:
        pygame.draw.circle(surf, C_META, (160, panel_y+22), 8)
        surf.blit(font_small.render(f"Meta: {met}", True, C_ROJO), (176, panel_y+14))

    surf.blit(font_msg.render(msg, True, C_GRIS),
              font_msg.render(msg, True, C_GRIS).get_rect(centerx=ANCHO//2, y=panel_y+46))
    ayuda_txt = "inicio <N>  |  meta <N>  |  empezar  |  reiniciar  |  nodos  |  salir"
    surf.blit(font_small.render(ayuda_txt, True, (75, 75, 105)),
              font_small.render(ayuda_txt, True, (75, 75, 105)).get_rect(centerx=ANCHO//2, y=panel_y+68))


# ══════════════════════════════════════════════════════════════════════════════
#  ANIMACION DE DIJKSTRA  (corre en hilo secundario)
# ══════════════════════════════════════════════════════════════════════════════

def _set_estado(nodo, color):
    with lock:
        estados[nodo] = color

def _set_mensaje(txt):
    global mensaje
    with lock:
        mensaje = txt

def dijkstra_animado(origen, destino):
    """
    Usa g.shortest_distances() para calcular distancias y padres,
    pero re-implementa el recorrido paso a paso para poder animar
    cada nodo visitado en tiempo real.
    """
    global en_proceso, ruta_linea, robot_pos

    dist    = {n: float("inf") for n in g.graph}
    parents = {n: None for n in g.graph}
    dist[origen] = 0

    pq = [(0, origen)]
    heapify(pq)
    visited = set()

    _set_mensaje(f"Explorando el grafo desde  {origen}  →  {destino} ...")

    while pq:
        current_distance, current_node = heappop(pq)

        if current_node in visited:
            continue
        visited.add(current_node)

        # Colorear nodo visitado en pantalla
        if current_node != origen and current_node != destino:
            _set_estado(current_node, C_VISITADO)

        time.sleep(0.22)   # pausa para que se vea la animacion

        if current_node == destino:
            break

        for neighbor, weight in g.graph[current_node]:
            if neighbor in visited:
                continue
            distance = current_distance + weight
            if distance < dist[neighbor]:
                dist[neighbor]    = distance
                parents[neighbor] = current_node
                heappush(pq, (distance, neighbor))

    # ── Reconstruir ruta usando parents ──────────────────────────────────────
    path = []
    cur  = destino
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()

    if not path or path[0] != origen:
        _set_mensaje("No existe ruta entre los nodos seleccionados.")
        print("\n[!] No existe ruta.\n>> ", end="", flush=True)
        en_proceso = False
        return

    # Colorear nodos de la ruta
    for nodo in path:
        if nodo != origen and nodo != destino:
            _set_estado(nodo, C_RUTA)

    # Construir segmentos de linea para la ruta
    segs = []
    for i in range(len(path)-1):
        a, b = _segmento_entre_nodos(NODOS[path[i]], NODOS[path[i+1]])
        segs.append((*a, *b))
    with lock:
        ruta_linea = segs

    time.sleep(0.15)

    # Animar robot recorriendo la ruta
    for nodo in path:
        with lock:
            robot_pos = NODOS[nodo]
        time.sleep(0.35)

    with lock:
        robot_pos = None

    costo    = dist[destino]
    ruta_str = " → ".join(path)
    _set_mensaje(f"Ruta: {ruta_str}   |   Costo: {costo}   |   Escribe 'reiniciar'")
    print(f"\n[✓] Ruta: {ruta_str}")
    print(f"    Costo total: {costo}\n>> ", end="", flush=True)
    en_proceso = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLA  (hilo secundario — lee input del usuario)
# ══════════════════════════════════════════════════════════════════════════════

AYUDA = """
Comandos disponibles:
  inicio <NODO>   → nodo de partida     ej: inicio A
  meta <NODO>     → nodo de destino     ej: meta O
  empezar         → ejecuta Dijkstra
  reiniciar       → limpia todo
  nodos           → lista los nodos del grafo
  ayuda           → muestra este mensaje
  salir           → cierra el programa
"""

def consola():
    global inicio, meta, en_proceso, estados, ruta_linea, robot_pos, mensaje, corriendo

    nodos_str = "  ".join(sorted(NODOS.keys()))
    print(AYUDA)
    print(f"Nodos disponibles: {nodos_str}\n")

    while corriendo:
        try:
            entrada = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            corriendo = False
            break

        if not entrada:
            continue

        partes = entrada.split()
        cmd    = partes[0].lower()

        # ── inicio ────────────────────────────────────────────────────────────
        if cmd == "inicio":
            if en_proceso:
                print("[!] Espera a que termine la animacion."); continue
            if len(partes) < 2:
                print("[!] Uso:  inicio <NODO>   ej: inicio A"); continue
            n = partes[1].upper()
            if n not in NODOS:
                print(f"[!] Nodo invalido. Validos: {nodos_str}"); continue
            if n == meta:
                print("[!] Ese nodo ya es la Meta, elige otro."); continue
            with lock:
                inicio = n
            suf = f"   Meta: {meta}   — escribe 'empezar'" if meta else "   — escribe:  meta <NODO>"
            _set_mensaje(f"Inicio: {n}" + suf)
            print(f"[✓] Inicio = {n}" + (f"   Meta = {meta}   →  escribe 'empezar'" if meta else "   →  escribe:  meta <NODO>"))

        # ── meta ──────────────────────────────────────────────────────────────
        elif cmd == "meta":
            if en_proceso:
                print("[!] Espera a que termine la animacion."); continue
            if len(partes) < 2:
                print("[!] Uso:  meta <NODO>   ej: meta O"); continue
            n = partes[1].upper()
            if n not in NODOS:
                print(f"[!] Nodo invalido. Validos: {nodos_str}"); continue
            if n == inicio:
                print("[!] Ese nodo ya es el Inicio, elige otro."); continue
            with lock:
                meta = n
            suf = f"   Inicio: {inicio}   — escribe 'empezar'" if inicio else "   — escribe:  inicio <NODO>"
            _set_mensaje(f"Meta: {n}" + suf)
            print(f"[✓] Meta = {n}" + (f"   Inicio = {inicio}   →  escribe 'empezar'" if inicio else "   →  escribe:  inicio <NODO>"))

        # ── empezar ───────────────────────────────────────────────────────────
        elif cmd == "empezar":
            if en_proceso:
                print("[!] Ya hay una animacion en curso."); continue
            with lock:
                ini, met = inicio, meta
            if not ini:
                print("[!] Falta el Inicio.  Escribe:  inicio <NODO>"); continue
            if not met:
                print("[!] Falta la Meta.  Escribe:  meta <NODO>"); continue
            en_proceso = True
            with lock:
                estados.clear(); ruta_linea = []; robot_pos = None
            print(f"[→] Ejecutando Dijkstra de {ini} a {met}...")
            threading.Thread(
                target=dijkstra_animado, args=(ini, met), daemon=True
            ).start()

        # ── reiniciar ─────────────────────────────────────────────────────────
        elif cmd == "reiniciar":
            if en_proceso:
                print("[!] Espera a que termine la animacion."); continue
            with lock:
                inicio = meta = None
                estados.clear(); ruta_linea = []; robot_pos = None
                mensaje = "Listo para una nueva busqueda."
            print("[✓] Reiniciado.")

        # ── nodos ─────────────────────────────────────────────────────────────
        elif cmd == "nodos":
            print(f"Nodos disponibles: {nodos_str}")

        # ── ayuda ─────────────────────────────────────────────────────────────
        elif cmd in ("ayuda", "help", "?"):
            print(AYUDA)

        # ── salir ─────────────────────────────────────────────────────────────
        elif cmd == "salir":
            print("Cerrando..."); corriendo = False; break

        else:
            print(f"[?] Comando desconocido: '{cmd}'.  Escribe 'ayuda'.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP DE PYGAME
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    surf  = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Dijkstra con Pygame  —  controla desde la consola")
    clock = pygame.time.Clock()

    font_nodo  = pygame.font.SysFont("Arial", 14, bold=True)
    font_peso  = pygame.font.SysFont("Arial", 10)
    font_msg   = pygame.font.SysFont("Arial", 13)
    font_small = pygame.font.SysFont("Arial", 12)

    threading.Thread(target=consola, daemon=True).start()

    global corriendo
    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

        surf.fill(C_FONDO)
        dibujar_grafo(surf, font_peso, font_nodo)
        dibujar_panel(surf, font_msg, font_small)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
