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
"""

import pygame
import heapq
import time
import threading
import sys

# ── Configuracion ─────────────────────────────────────────────────────────────

ANCHO, ALTO = 960, 680
FPS         = 60
RADIO       = 24

# Colores (RGB)
C_FONDO     = ( 20,  20,  46)
C_ARISTA    = ( 74,  74, 106)
C_NODO      = ( 45,  45,  78)
C_BORDE     = (127, 119, 221)
C_TEXTO     = (224, 224, 240)
C_INICIO    = ( 29, 158, 117)
C_META      = (216,  90,  48)
C_VISITADO  = ( 83,  74, 183)
C_RUTA      = (239, 159,  39)
C_PESO      = (159, 225, 203)
C_PANEL     = ( 12,  12,  30)
C_BLANCO    = (255, 255, 255)
C_GRIS      = (150, 150, 150)
C_VERDE     = ( 29, 158, 117)
C_ROJO      = (216,  90,  48)

# ── Nodos y aristas ───────────────────────────────────────────────────────────

NODOS = {
    "A": (120, 150), "B": (260, 80),  "C": (400,  80), "D": (540,  80),
    "E": (680, 150), "F": (120, 280), "G": (260, 260), "H": (400, 280),
    "I": (540, 260), "J": (680, 280), "K": (120, 420), "L": (260, 450),
    "M": (400, 480), "N": (540, 450), "O": (680, 420),
}

ARISTAS = [
    ("A","B", 4), ("A","F", 3), ("B","C", 5), ("B","G", 7),
    ("C","D", 3), ("C","H", 4), ("D","E", 6), ("D","I", 5),
    ("E","J", 2), ("F","G", 6), ("F","K", 5), ("G","H", 3),
    ("G","L", 8), ("H","I", 4), ("H","M", 6), ("I","J", 3),
    ("I","N", 7), ("J","O", 4), ("K","L", 5), ("L","M", 3),
    ("L","N", 9), ("M","N", 4), ("N","O", 5), ("B","D", 9),
    ("F","H", 8), ("K","M", 7),
]

grafo = {n: [] for n in NODOS}
for u, v, w in ARISTAS:
    grafo[u].append((v, w))
    grafo[v].append((u, w))

# ── Estado compartido (protegido con lock) ────────────────────────────────────

lock       = threading.Lock()
inicio     = None
meta       = None
en_proceso = False
estados    = {}          # nodo -> color de relleno
mensaje    = "Escribe comandos en la consola  |  inicio <N>   meta <N>   empezar"
ruta_linea = []          # lista de pares (x1,y1,x2,y2) para la ruta
robot_pos  = None        # (x, y) posicion actual del robot

# ── Helpers de dibujo ─────────────────────────────────────────────────────────

def color_nodo(nombre):
    with lock:
        ini, met, est = inicio, meta, estados.copy()
    if nombre == ini: return C_INICIO
    if nombre == met: return C_META
    return est.get(nombre, C_NODO)

def dibujar_flecha_redondeada(surf, color, p1, p2, grosor=2):
    """Dibuja una linea entre dos nodos dejando espacio para el circulo."""
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    dist   = (dx**2 + dy**2) ** 0.5
    if dist == 0: return
    ux, uy = dx/dist, dy/dist
    inicio_l = (int(p1[0] + ux*RADIO), int(p1[1] + uy*RADIO))
    fin_l    = (int(p2[0] - ux*RADIO), int(p2[1] - uy*RADIO))
    pygame.draw.line(surf, color, inicio_l, fin_l, grosor)

def dibujar_grafo(surf, font_peso, font_nodo):
    # Aristas base
    for u, v, w in ARISTAS:
        p1, p2 = NODOS[u], NODOS[v]
        dibujar_flecha_redondeada(surf, C_ARISTA, p1, p2, 2)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        txt = font_peso.render(str(w), True, C_PESO)
        surf.blit(txt, txt.get_rect(center=(mx, my-10)))

    # Ruta resaltada
    with lock:
        rl = ruta_linea.copy()
    for x1,y1,x2,y2 in rl:
        pygame.draw.line(surf, C_RUTA, (x1,y1), (x2,y2), 5)

    # Nodos
    for nombre, (x, y) in NODOS.items():
        fill  = color_nodo(nombre)
        borde = C_BORDE
        if fill == C_INICIO: borde = (15, 110, 86)
        if fill == C_META:   borde = (153, 60, 29)
        if fill == C_RUTA:   borde = (133, 79, 11)
        pygame.draw.circle(surf, fill,  (x, y), RADIO)
        pygame.draw.circle(surf, borde, (x, y), RADIO, 2)
        txt = font_nodo.render(nombre, True, C_TEXTO)
        surf.blit(txt, txt.get_rect(center=(x, y)))

    # Robot
    with lock:
        rp = robot_pos
    if rp:
        pygame.draw.circle(surf, C_RUTA, rp, 10)
        pygame.draw.circle(surf, C_BLANCO, rp, 10, 2)

def dibujar_panel(surf, font_msg, font_small):
    panel_y = ALTO - 90
    pygame.draw.rect(surf, C_PANEL, (0, panel_y, ANCHO, 90))
    pygame.draw.line(surf, C_BORDE, (0, panel_y), (ANCHO, panel_y), 1)

    with lock:
        msg = mensaje
        ini = inicio
        met = meta

    # Indicadores inicio / meta
    if ini:
        pygame.draw.circle(surf, C_INICIO, (30, panel_y+22), 8)
        t = font_small.render(f"Inicio: {ini}", True, C_VERDE)
        surf.blit(t, (46, panel_y+14))
    if met:
        pygame.draw.circle(surf, C_META, (160, panel_y+22), 8)
        t = font_small.render(f"Meta: {met}", True, C_ROJO)
        surf.blit(t, (176, panel_y+14))

    # Mensaje principal
    t = font_msg.render(msg, True, C_GRIS)
    surf.blit(t, t.get_rect(centerx=ANCHO//2, y=panel_y+48))

    # Instrucciones
    t2 = font_small.render("Escribe en la consola  →  inicio <N>  |  meta <N>  |  empezar  |  reiniciar  |  salir", True, (80,80,110))
    surf.blit(t2, t2.get_rect(centerx=ANCHO//2, y=panel_y+70))

# ── Dijkstra en hilo secundario ───────────────────────────────────────────────

def set_estado(nodo, color):
    with lock:
        estados[nodo] = color

def set_mensaje(txt):
    global mensaje
    with lock:
        mensaje = txt

def dijkstra_thread(origen, destino):
    global en_proceso, ruta_linea, robot_pos

    dist = {n: float("inf") for n in NODOS}
    prev = {n: None for n in NODOS}
    dist[origen] = 0
    pq = [(0, origen)]
    visitados = set()

    set_mensaje(f"Buscando ruta de  {origen}  →  {destino} ...")

    while pq:
        d, u = heapq.heappop(pq)
        if u in visitados: continue
        visitados.add(u)

        if u != origen and u != destino:
            set_estado(u, C_VISITADO)
        time.sleep(0.22)

        if u == destino: break

        for v, w in grafo[u]:
            if v in visitados: continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruir ruta
    ruta = []; cur = destino
    while cur: ruta.append(cur); cur = prev[cur]
    ruta.reverse()

    if ruta[0] != origen:
        set_mensaje("No existe ruta entre los nodos seleccionados.")
        print("\n[!] No existe ruta.\n>> ", end="", flush=True)
        en_proceso = False
        return

    # Colorear ruta
    for nodo in ruta:
        if nodo != origen and nodo != destino:
            set_estado(nodo, C_RUTA)

    # Construir segmentos de linea
    segmentos = []
    for i in range(len(ruta)-1):
        x1,y1 = NODOS[ruta[i]]; x2,y2 = NODOS[ruta[i+1]]
        dx,dy  = x2-x1, y2-y1
        dist_s = (dx**2+dy**2)**0.5
        ux,uy  = dx/dist_s, dy/dist_s
        segmentos.append((
            int(x1+ux*RADIO), int(y1+uy*RADIO),
            int(x2-ux*RADIO), int(y2-uy*RADIO)
        ))
    with lock:
        ruta_linea = segmentos

    time.sleep(0.1)

    # Animar robot
    for nodo in ruta:
        with lock:
            robot_pos = NODOS[nodo]
        time.sleep(0.35)

    with lock:
        robot_pos = None

    costo    = dist[destino]
    ruta_str = " → ".join(ruta)
    set_mensaje(f"Ruta: {ruta_str}   |   Costo total: {costo}   |   Escribe 'reiniciar'")
    print(f"\n[✓] Ruta: {ruta_str}   Costo: {costo}\n>> ", end="", flush=True)
    en_proceso = False

# ── Consola en hilo secundario ────────────────────────────────────────────────

AYUDA = """
Comandos disponibles:
  inicio <NODO>   → nodo de partida     ej: inicio A
  meta <NODO>     → nodo de destino     ej: meta O
  empezar         → ejecuta Dijkstra
  reiniciar       → limpia todo
  nodos           → lista los nodos
  ayuda           → muestra este mensaje
  salir           → cierra el programa
"""

corriendo = True

def consola():
    global inicio, meta, en_proceso, estados, ruta_linea, robot_pos, corriendo, mensaje

    nodos_str = "  ".join(sorted(NODOS.keys()))
    print(AYUDA)
    print(f"Nodos: {nodos_str}\n")

    while corriendo:
        try:
            entrada = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            corriendo = False; break

        if not entrada: continue
        partes = entrada.split()
        cmd    = partes[0].lower()

        if cmd == "inicio":
            if en_proceso:
                print("[!] Espera a que termine la animacion."); continue
            if len(partes) < 2:
                print("[!] Uso:  inicio <NODO>   ej: inicio A"); continue
            n = partes[1].upper()
            if n not in NODOS:
                print(f"[!] Nodo invalido. Validos: {nodos_str}"); continue
            if n == meta:
                print("[!] Ese nodo ya es la Meta."); continue
            with lock:
                inicio = n
            set_mensaje(f"Inicio: {n}" + (f"   Meta: {meta}   — escribe 'empezar'" if meta else "   — escribe:  meta <NODO>"))
            print(f"[✓] Inicio = {n}" + (f"   Meta = {meta}   →  escribe 'empezar'" if meta else "   →  escribe:  meta <NODO>"))

        elif cmd == "meta":
            if en_proceso:
                print("[!] Espera a que termine la animacion."); continue
            if len(partes) < 2:
                print("[!] Uso:  meta <NODO>   ej: meta O"); continue
            n = partes[1].upper()
            if n not in NODOS:
                print(f"[!] Nodo invalido. Validos: {nodos_str}"); continue
            if n == inicio:
                print("[!] Ese nodo ya es el Inicio."); continue
            with lock:
                meta = n
            set_mensaje(f"Meta: {n}" + (f"   Inicio: {inicio}   — escribe 'empezar'" if inicio else "   — escribe:  inicio <NODO>"))
            print(f"[✓] Meta = {n}" + (f"   Inicio = {inicio}   →  escribe 'empezar'" if inicio else "   →  escribe:  inicio <NODO>"))

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
            threading.Thread(target=dijkstra_thread, args=(ini, met), daemon=True).start()

        elif cmd == "reiniciar":
            if en_proceso:
                print("[!] Espera a que termine la animacion."); continue
            with lock:
                inicio = meta = None
                estados.clear(); ruta_linea = []; robot_pos = None
                mensaje = "Listo para una nueva busqueda."
            print("[✓] Reiniciado.")

        elif cmd == "nodos":
            print(f"Nodos: {nodos_str}")

        elif cmd in ("ayuda","help","?"):
            print(AYUDA)

        elif cmd == "salir":
            print("Cerrando..."); corriendo = False; break

        else:
            print(f"[?] Comando desconocido: '{cmd}'.  Escribe 'ayuda'.")

# ── Main loop de Pygame ───────────────────────────────────────────────────────

def main():
    pygame.init()
    surf  = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Dijkstra con Pygame  —  controla desde la consola")
    clock = pygame.time.Clock()

    font_nodo  = pygame.font.SysFont("Arial", 14, bold=True)
    font_peso  = pygame.font.SysFont("Arial", 10)
    font_msg   = pygame.font.SysFont("Arial", 13)
    font_small = pygame.font.SysFont("Arial", 12)

    # Arrancar consola en hilo separado
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
