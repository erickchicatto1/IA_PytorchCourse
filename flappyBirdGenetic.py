import pygame
import random
import time
import math
from pygame.locals import *

#https://www.datacamp.com/es/tutorial/genetic-algorithm-python

# ─────────────────────────────────────────────
#  VARIABLES
# ─────────────────────────────────────────────
SCREEN_WIDTH  = 400
SCREEN_HEIGHT = 600
SPEED         = 20
GRAVITY       = 2.5
GAME_SPEED    = 15

GROUND_WIDTH  = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH  = 80
PIPE_HEIGHT = 500
PIPE_GAP    = 150

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.mixer.init()

# ─────────────────────────────────────────────
#  RED NEURONAL (desde cero, sin librerías extra)
#  Arquitectura: 4 entradas → 6 ocultas → 1 salida
#  Entrenamiento: Algoritmo genético (neuroevolución)
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

class NeuralNetwork:
    """Red neuronal simple: 4-6-1."""

    INPUT  = 4
    HIDDEN = 6
    OUTPUT = 1

    def __init__(self):
        # Pesos capa oculta: INPUT x HIDDEN
        self.w1 = [[random.gauss(0, 1) for _ in range(self.HIDDEN)]
                   for _ in range(self.INPUT)]
        self.b1 = [random.gauss(0, 1) for _ in range(self.HIDDEN)]
        # Pesos capa salida: HIDDEN x OUTPUT
        self.w2 = [[random.gauss(0, 1) for _ in range(self.OUTPUT)]
                   for _ in range(self.HIDDEN)]
        self.b2 = [random.gauss(0, 1) for _ in range(self.OUTPUT)]

    def forward(self, inputs):
        # Capa oculta
        hidden = []
        for j in range(self.HIDDEN):
            val = self.b1[j]
            for i in range(self.INPUT):
                val += inputs[i] * self.w1[i][j]
            hidden.append(sigmoid(val))
        # Capa de salida
        output = []
        for j in range(self.OUTPUT):
            val = self.b2[j]
            for i in range(self.HIDDEN):
                val += hidden[i] * self.w2[i][j]
            output.append(sigmoid(val))
        return output

    def copy(self):
        child = NeuralNetwork()
        child.w1 = [row[:] for row in self.w1]
        child.b1 = self.b1[:]
        child.w2 = [row[:] for row in self.w2]
        child.b2 = self.b2[:]
        return child

    def mutate(self, rate=0.15, strength=0.5):
        def _m(v):
            return v + random.gauss(0, strength) if random.random() < rate else v
        self.w1 = [[_m(v) for v in row] for row in self.w1]
        self.b1 = [_m(v) for v in self.b1]
        self.w2 = [[_m(v) for v in row] for row in self.w2]
        self.b2 = [_m(v) for v in self.b2]

    def crossover(self, other):
        child = self.copy()
        for i in range(self.INPUT):
            for j in range(self.HIDDEN):
                if random.random() < 0.5:
                    child.w1[i][j] = other.w1[i][j]
        for j in range(self.HIDDEN):
            if random.random() < 0.5:
                child.b1[j] = other.b1[j]
        for i in range(self.HIDDEN):
            for j in range(self.OUTPUT):
                if random.random() < 0.5:
                    child.w2[i][j] = other.w2[i][j]
        for j in range(self.OUTPUT):
            if random.random() < 0.5:
                child.b2[j] = other.b2[j]
        return child

# ─────────────────────────────────────────────
#  SPRITES
# ─────────────────────────────────────────────

class Bird(pygame.sprite.Sprite):
    def __init__(self, brain=None):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha(),
        ]
        self.speed         = SPEED
        self.current_image = 0
        self.image         = self.images[0]
        self.mask          = pygame.mask.from_surface(self.image)
        self.rect          = self.image.get_rect()
        self.rect[0]       = SCREEN_WIDTH / 6
        self.rect[1]       = SCREEN_HEIGHT / 2
        self.alive         = True
        self.score         = 0          # frames sobrevividos
        self.brain         = brain if brain else NeuralNetwork()

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image         = self.images[self.current_image]
        self.speed        += GRAVITY
        self.rect[1]      += self.speed
        self.score        += 1

    def bump(self):
        self.speed = -SPEED

    def think(self, pipes):
        """Decide si saltar basándose en la red neuronal."""
        # Buscar el siguiente par de tuberías (la que está más adelante del pájaro)
        next_pipe = None
        next_pipe_inv = None
        min_dist = float('inf')

        sprites = pipes.sprites()
        # Los pipes vienen en pares (normal, invertido)
        for i in range(0, len(sprites) - 1, 2):
            p = sprites[i]
            dist = p.rect[0] - self.rect[0]
            if dist > -PIPE_WIDTH and dist < min_dist:
                min_dist      = dist
                next_pipe     = sprites[i]
                next_pipe_inv = sprites[i + 1]

        if next_pipe is None:
            return  # sin tubería visible → no hacer nada

        # ── Entradas normalizadas ──────────────────────────
        bird_y          = self.rect[1] / SCREEN_HEIGHT          # posición vertical
        bird_vel        = self.speed   / 20.0                    # velocidad vertical
        pipe_dist       = min_dist     / SCREEN_WIDTH            # distancia horizontal
        # Centro del hueco entre las dos tuberías
        gap_center_y    = (next_pipe.rect[1] + next_pipe.rect[3] +
                           next_pipe_inv.rect[1]) / 2.0
        gap_center_norm = gap_center_y / SCREEN_HEIGHT           # posición del hueco

        inputs = [bird_y, bird_vel, pipe_dist, gap_center_norm]
        output = self.brain.forward(inputs)

        if output[0] > 0.5:
            self.bump()


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image   = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED

# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    size         = random.randint(100, 300)
    pipe         = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True,  xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


def create_population(size, brain=None):
    group = []
    for _ in range(size):
        b = Bird(brain.copy() if brain else None)
        if brain:
            b.brain.mutate()
        group.append(b)
    return group


def next_generation(population, pop_size):
    """Selección elitista + cruce + mutación."""
    population.sort(key=lambda b: b.score, reverse=True)
    elite = population[:2]                            # los 2 mejores sobreviven
    children = []
    for e in elite:
        c = Bird(e.brain.copy())
        children.append(c)
    while len(children) < pop_size:
        # Torneo entre los top-5
        candidates = population[:min(5, len(population))]
        p1 = random.choice(candidates)
        p2 = random.choice(candidates)
        child_brain = p1.brain.crossover(p2.brain)
        child_brain.mutate()
        children.append(Bird(child_brain))
    return children

# ─────────────────────────────────────────────
#  CONFIGURACIÓN PYGAME
# ─────────────────────────────────────────────

POP_SIZE = 20          # pájaros por generación
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird — IA Neuroevolución')

BACKGROUND  = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND  = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

font_large = pygame.font.SysFont('Arial', 22, bold=True)
font_small = pygame.font.SysFont('Arial', 16)

clock = pygame.time.Clock()

# ─────────────────────────────────────────────
#  PANTALLA DE INICIO
# ─────────────────────────────────────────────

waiting = True
while waiting:
    clock.tick(15)
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()
        if event.type == KEYDOWN:
            if event.key in (K_SPACE, K_UP, K_RETURN):
                waiting = False

    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))
    hint = font_small.render('Presiona ESPACIO para iniciar la IA', True, (255, 255, 255))
    screen.blit(hint, (SCREEN_WIDTH // 2 - hint.get_width() // 2, 480))
    pygame.display.update()

# ─────────────────────────────────────────────
#  BUCLE PRINCIPAL — ENTRENAMIENTO
# ─────────────────────────────────────────────

population    = create_population(POP_SIZE)
generation    = 1
best_score    = 0
best_ever     = 0

running = True
while running:
    # ── Crear grupos de suelo y tuberías frescos ──
    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground_group.add(Ground(GROUND_WIDTH * i))

    pipe_group = pygame.sprite.Group()
    for i in range(2):
        pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    # Grupo de pájaros vivos
    bird_group = pygame.sprite.Group()
    alive      = list(population)
    for b in alive:
        b.rect[1] = SCREEN_HEIGHT / 2
        b.speed   = SPEED
        b.alive   = True
        b.score   = 0
        bird_group.add(b)

    gen_running = True
    while gen_running:
        clock.tick(30)     # 30 FPS (puedes subir para entrenar más rápido)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                gen_running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                    gen_running = False

        # ── Actualizar suelo ──────────────────────────
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDTH - 20))

        # ── Actualizar tuberías ───────────────────────
        if pipe_group.sprites() and is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDTH * 2)
            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        ground_group.update()
        pipe_group.update()

        # ── IA: cada pájaro decide ────────────────────
        for bird in list(bird_group.sprites()):
            bird.think(pipe_group)

        bird_group.update()

        # ── Colisiones ───────────────────────────────
        dead_ground = pygame.sprite.groupcollide(
            bird_group, ground_group, False, False,
            pygame.sprite.collide_mask)
        dead_pipe = pygame.sprite.groupcollide(
            bird_group, pipe_group, False, False,
            pygame.sprite.collide_mask)

        for bird in list(bird_group.sprites()):
            # Colisión con suelo / tuberías o salió por arriba
            if bird in dead_ground or bird in dead_pipe or bird.rect[1] < -50:
                bird.alive = False
                bird_group.remove(bird)

        if not bird_group:
            gen_running = False   # toda la generación murió

        # ── Dibujar ──────────────────────────────────
        screen.blit(BACKGROUND, (0, 0))
        pipe_group.draw(screen)
        ground_group.draw(screen)
        bird_group.draw(screen)

        # ── HUD ──────────────────────────────────────
        alive_count = len(bird_group)
        best_score  = max((b.score for b in population), default=0)
        best_ever   = max(best_ever, best_score)

        pygame.draw.rect(screen, (0, 0, 0, 160), (5, 5, 200, 90))
        screen.blit(font_large.render(f'Generación: {generation}',    True, (255, 230, 0)),   (10, 10))
        screen.blit(font_small.render(f'Vivos:  {alive_count}/{POP_SIZE}', True, (200, 255, 200)), (10, 38))
        screen.blit(font_small.render(f'Mejor esta gen: {best_score}',     True, (180, 220, 255)), (10, 58))
        screen.blit(font_small.render(f'Récord total:   {best_ever}',      True, (255, 180, 100)), (10, 78))

        pygame.display.update()

    # ── Nueva generación ──────────────────────────
    if not running:
        break

    population  = next_generation(population, POP_SIZE)
    generation += 1

pygame.quit()
