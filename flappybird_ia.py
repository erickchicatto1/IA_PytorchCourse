import pygame, random, time, math
from pygame.locals import *

# ─────────────────────────────────────────
#  CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────
SCREEN_WIDTH  = 400
SCREEN_HEIGHT = 600
SPEED         = 20
GRAVITY       = 2.5
GAME_SPEED    = 15

GROUND_WIDTH  = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH    = 80
PIPE_HEIGHT   = 500
PIPE_GAP      = 150

# IA / Evolución
POPULATION    = 20
MUTATION_RATE = 0.1
MUTATION_STR  = 0.3

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.mixer.init()
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird – IA')

BACKGROUND  = pygame.transform.scale(
    pygame.image.load('assets/sprites/background-day.png'),
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

font_small = pygame.font.SysFont('consolas', 14)

# ═══════════════════════════════════════════
#  RED NEURONAL
# ═══════════════════════════════════════════
class NeuralNetwork:
    def __init__(self, weights=None):
        self.weights = weights[:] if weights else [random.uniform(-1,1) for _ in range(31)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(-500, min(500, x))))

    def predict(self, inputs):
        w = self.weights
        hidden = []
        for i in range(6):
            z = w[i*3]*inputs[0] + w[i*3+1]*inputs[1] + w[i*3+2]*inputs[2] + w[18+i]
            hidden.append(self.sigmoid(z))
        z_out = sum(hidden[i]*w[24+i] for i in range(6)) + w[30]
        return self.sigmoid(z_out)

    def mutate(self):
        child = NeuralNetwork(self.weights)
        for i in range(len(child.weights)):
            if random.random() < MUTATION_RATE:
                child.weights[i] += random.uniform(-MUTATION_STR, MUTATION_STR)
        return child

    def crossover(self, other):
        point = random.randint(1, len(self.weights)-1)
        return NeuralNetwork(self.weights[:point] + other.weights[point:])


# ═══════════════════════════════════════════
#  SPRITES
# ═══════════════════════════════════════════
class Bird(pygame.sprite.Sprite):
    def __init__(self, brain=None):
        super().__init__()
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha(),
        ]
        self.current_image = 0
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH // 6
        self.rect[1] = SCREEN_HEIGHT // 2

        self.speed = SPEED
        self.alive = True
        self.fitness = 0
        self.brain = brain if brain else NeuralNetwork()

    def update(self):
        if not self.alive:
            return
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed
        self.fitness += 1

    def bump(self):
        self.speed = -SPEED

    def think(self, pipes):
        nearest = None
        min_dist = float('inf')
        for pipe in pipes:
            if not pipe.inverted and pipe.rect[0] > self.rect[0]:
                dist = pipe.rect[0] - self.rect[0]
                if dist < min_dist:
                    min_dist = dist
                    nearest = pipe

        if nearest is None:
            return

        top_y = nearest.partner_y

        dist_x = min_dist / SCREEN_WIDTH
        dist_y = (self.rect[1] - top_y) / SCREEN_HEIGHT
        speed_n = self.speed / 20

        if self.brain.predict([dist_x, dist_y, speed_n]) > 0.5:
            self.bump()

    def die(self):
        self.alive = False


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        super().__init__()
        self.image = pygame.transform.scale(
            pygame.image.load('assets/sprites/pipe-green.png').convert_alpha(),
            (PIPE_WIDTH, PIPE_HEIGHT)
        )
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.inverted = inverted

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        super().__init__()
        self.image = pygame.transform.scale(
            pygame.image.load('assets/sprites/base.png').convert_alpha(),
            (GROUND_WIDTH, GROUND_HEIGHT)
        )
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


# ═══════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════
def is_off_screen(sprite):
    return sprite.rect[0] < -sprite.rect[2]


def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inv = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)

    pipe.partner_y = pipe_inv.rect[1] + PIPE_HEIGHT
    pipe_inv.partner_y = pipe.partner_y

    return pipe, pipe_inv


# ═══════════════════════════════════════════
#  PANTALLA INICIAL (FIX AQUÍ)
# ═══════════════════════════════════════════
def begin_screen():
    bird = Bird()
    bird_group = pygame.sprite.Group(bird)
    ground_group = pygame.sprite.Group(Ground(0), Ground(GROUND_WIDTH))

    clock = pygame.time.Clock()

    while True:
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); exit()
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                return

        screen.blit(BACKGROUND, (0, 0))

        bird_group.update()   # ← FIX
        ground_group.update()

        bird_group.draw(screen)
        ground_group.draw(screen)

        pygame.display.update()


# ═══════════════════════════════════════════
#  LOOP PRINCIPAL
# ═══════════════════════════════════════════
def run_ai():
    birds = [Bird() for _ in range(POPULATION)]

    while True:
        pipe_group = pygame.sprite.Group()
        for i in range(2):
            p = get_random_pipes(600 + i*300)
            pipe_group.add(p[0], p[1])

        for b in birds:
            b.alive = True
            b.fitness = 0
            b.rect[1] = SCREEN_HEIGHT // 2
            b.speed = SPEED

        clock = pygame.time.Clock()
        running = True

        while running:
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit(); exit()

            alive = [b for b in birds if b.alive]

            for b in alive:
                b.think(pipe_group.sprites())

            for b in birds:
                b.update()

            pipe_group.update()

            for b in alive:
                if b.rect[1] > SCREEN_HEIGHT or b.rect[1] < 0:
                    b.die()

            screen.blit(BACKGROUND, (0, 0))
            pipe_group.draw(screen)

            for b in birds:
                if b.alive:
                    screen.blit(b.image, b.rect)

            pygame.display.update()

            if not alive:
                running = False

        birds = [Bird() for _ in range(POPULATION)]


# ═══════════════════════════════════════════
begin_screen()
run_ai()
