import os
import time
import math
import pygame
import random
import cv2
import numpy as np
from keras.models import load_model

# -----------------------------
# CONFIGURATION
# -----------------------------
INITIAL_SCROLL_SPEED = 7
MAX_SCROLL_SPEED = 10
INITIAL_HOLE_PROB = 0.1
MAX_HOLE_PROB = 0.5
DISTANCE_FOR_MAX_DIFFICULTY = 2000

HOLE_WIDTH_MIN = 120
HOLE_WIDTH_MAX = 150
SAFE_GROUND_LENGTH = 5
GROUND_HEIGHT = 100
BLOCK_WIDTH = 100

BIRD_SIZE = 40
GRAVITY = 1.0
JUMP_STRENGTH = -9

NUM_CLOUDS = 5
NUM_SKY_DOTS = 80
CLOUD_PARALLAX = 0.3

# 8-bit style sky gradient top to bottom
SKY_COLOR_TOP = (20, 40, 90)     # darker top
SKY_COLOR_BOTTOM = (70, 120, 180)  # lighter bottom

# We'll use SKY_COLOR_BOTTOM as the color to "carve out" the hole area
# so the hole looks like the bottom color of the sky.
HOLE_COLOR = SKY_COLOR_BOTTOM

# Ground is just brown + green top (no gradient)
BROWN = (139, 69, 19)
GREEN = (76, 187, 23)

# Basic colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

GAME_WIDTH = 800
GAME_HEIGHT = 600

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
pygame.init()

# -----------------------------
# SOUND SETUP
# -----------------------------
pygame.mixer.init()
jump_sound = pygame.mixer.Sound("audiomass-output.mp3")
jump_channel = pygame.mixer.Channel(0)  # ensures new jump cuts off old sound

# Fullscreen letterbox
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
pygame.display.set_caption("Screaming Bird")

# Virtual surface (800×600)
game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))

# Fonts
menu_font = pygame.font.SysFont("Courier New", 36, bold=True)
score_font = pygame.font.SysFont("Courier New", 24, bold=True)

# -----------------------------
# LOAD AI MODEL & LABELS
# -----------------------------
model = load_model("keras_Model.h5", compile=False)
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

ai_prediction = ""
confidence_score = 0.0

# -----------------------------
# DIFFICULTY RAMPING
# -----------------------------
def update_difficulty(distance_travelled):
    ratio = min(distance_travelled / DISTANCE_FOR_MAX_DIFFICULTY, 1.0)
    scroll_speed = INITIAL_SCROLL_SPEED + (MAX_SCROLL_SPEED - INITIAL_SCROLL_SPEED) * ratio
    hole_probability = INITIAL_HOLE_PROB + (MAX_HOLE_PROB - INITIAL_HOLE_PROB) * ratio
    return scroll_speed, hole_probability

# -----------------------------
# DECORATIONS (Streetlights & Trees)
# -----------------------------
streetlights = []
trees = []
STREETLIGHT_CHANCE = 0.03
TREE_CHANCE = 0.15

def create_streetlight_surface():
    w, h = 36, 90
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, (70, 70, 80), (16, 30, 4, 50))   # post
    pygame.draw.rect(surf, (60, 60, 70), (14, 80, 8, 10))   # base
    pygame.draw.rect(surf, (90, 90, 100), (10, 25, 16, 5))  # lamp bottom
    pygame.draw.rect(surf, (180, 180, 100), (10, 10, 16, 15)) # lamp top
    pygame.draw.circle(surf, (255, 255, 150, 120), (18, 10), 8)
    return surf

def create_tree_surface():
    w, h = 72, 96
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, (80, 50, 20), (30, 60, 12, 36))   # trunk
    pygame.draw.rect(surf, GREEN, (10, 20, 52, 40))          # top rect
    for _ in range(4):
        rx = random.randint(10, 62)
        ry = random.randint(20, 60)
        rr = random.randint(8, 16)
        pygame.draw.circle(surf, (40, 200, 80), (rx, ry), rr)
    return surf

streetlight_surf = create_streetlight_surface()
tree_surf = create_tree_surface()

def place_decor(segment):
    if random.random() < STREETLIGHT_CHANCE:
        lx = segment.x + random.randint(0, BLOCK_WIDTH - 36)
        ly = segment.y - 90
        streetlights.append({"x": lx, "y": ly})

    if random.random() < TREE_CHANCE:
        tx = segment.x + random.randint(0, BLOCK_WIDTH - 72)
        ty = segment.y - 96
        trees.append({"x": tx, "y": ty})

def update_streetlights(scroll_speed):
    for s in streetlights:
        s["x"] -= scroll_speed
    streetlights[:] = [s for s in streetlights if s["x"] + 36 > 0]

def draw_streetlights():
    for s in streetlights:
        game_surface.blit(streetlight_surf, (s["x"], s["y"]))

def update_trees(scroll_speed):
    for t in trees:
        t["x"] -= scroll_speed
    trees[:] = [t for t in trees if t["x"] + 72 > 0]

def draw_trees():
    for t in trees:
        game_surface.blit(tree_surf, (t["x"], t["y"]))

# -----------------------------
# Bird Class (Two Sprites)
# -----------------------------
class Bird:
    def __init__(self):
        self.x = GAME_WIDTH // 4
        self.y = GAME_HEIGHT - GROUND_HEIGHT - BIRD_SIZE
        self.vel_y = 0
        self.on_ground = True

        self.image_walk = pygame.image.load("bird_walk.png").convert_alpha()
        self.image_jump = pygame.image.load("bird_jump.png").convert_alpha()

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_STRENGTH
            self.on_ground = False
            jump_channel.play(jump_sound)

    def update(self):
        self.vel_y += GRAVITY
        self.y += self.vel_y
        self.on_ground = False

        for segment in ground_segments:
            # lethal portion
            brown_rect = pygame.Rect(segment.x, segment.y + 10, BLOCK_WIDTH, GROUND_HEIGHT - 10)
            if brown_rect.colliderect(self.x, self.y, BIRD_SIZE, BIRD_SIZE):
                return False

            # safe top
            green_rect = pygame.Rect(segment.x, segment.y, BLOCK_WIDTH, 10)
            if green_rect.colliderect(self.x, self.y + BIRD_SIZE, BIRD_SIZE, 1):
                self.y = green_rect.top - BIRD_SIZE
                self.vel_y = 0
                self.on_ground = True
                break

        if self.y > GAME_HEIGHT:
            return False
        return True

    def draw(self):
        if self.on_ground:
            game_surface.blit(self.image_walk, (self.x, self.y))
        else:
            game_surface.blit(self.image_jump, (self.x, self.y))

# -----------------------------
# Clouds & Dots
# -----------------------------
clouds = []
dots = []

def create_cloud_surface(w, h):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for _ in range(random.randint(4, 6)):
        shade = random.randint(200, 255)
        rect_color = (shade, shade, shade)
        rw = random.randint(int(w*0.3), w)
        rh = random.randint(int(h*0.3), h)
        rx = random.randint(0, w - rw)
        ry = random.randint(0, h - rh)
        pygame.draw.rect(surf, rect_color, (rx, ry, rw, rh))
    return surf

def generate_sky():
    for _ in range(NUM_CLOUDS):
        w = random.randint(40, 100)
        h = random.randint(20, 50)
        x = random.randint(0, GAME_WIDTH - w)
        y = random.randint(0, 120)
        cloud_surf = create_cloud_surface(w, h)
        speed = CLOUD_PARALLAX
        clouds.append({"x": x, "y": y, "surf": cloud_surf, "speed": speed})

    for _ in range(NUM_SKY_DOTS):
        x = random.randint(0, GAME_WIDTH)
        y = random.randint(0, 150)
        dots.append((x, y))

def update_clouds(scroll_speed):
    for c in clouds:
        c["x"] -= c["speed"] * scroll_speed
        if c["x"] + c["surf"].get_width() < 0:
            c["x"] = GAME_WIDTH
            c["y"] = random.randint(0, 120)

def draw_clouds_and_dots():
    for d in dots:
        pygame.draw.rect(game_surface, YELLOW, (d[0], d[1], 2, 2))
    for c in clouds:
        game_surface.blit(c["surf"], (c["x"], c["y"]))

# -----------------------------
# 8-Bit Sky Gradient
# -----------------------------
def fill_sky_gradient():
    chunk_count = 16
    chunk_height = GAME_HEIGHT // chunk_count
    for i in range(chunk_count):
        alpha = i / (chunk_count - 1)
        r = int(SKY_COLOR_TOP[0] + alpha * (SKY_COLOR_BOTTOM[0] - SKY_COLOR_TOP[0]))
        g = int(SKY_COLOR_TOP[1] + alpha * (SKY_COLOR_BOTTOM[1] - SKY_COLOR_TOP[1]))
        b = int(SKY_COLOR_TOP[2] + alpha * (SKY_COLOR_BOTTOM[2] - SKY_COLOR_TOP[2]))
        color = (r, g, b)
        top_y = i * chunk_height
        pygame.draw.rect(game_surface, color, (0, top_y, GAME_WIDTH, chunk_height))

# -----------------------------
# Ground
# -----------------------------
ground_segments = []
holes = []

def create_ground_texture():
    """
    Solid brown ground with green top, plus random squares in the brown area.
    """
    tex = pygame.Surface((BLOCK_WIDTH, GROUND_HEIGHT))
    tex.fill(BROWN)

    # random squares in brown portion below top 10 px
    dot_color = (
        max(0, BROWN[0] - 30),
        max(0, BROWN[1] - 30),
        max(0, BROWN[2] - 30),
    )
    square_size = 8
    for y in range(10, GROUND_HEIGHT, square_size):
        for x in range(0, BLOCK_WIDTH, square_size):
            if random.random() < 0.3:
                pygame.draw.rect(tex, dot_color, (x, y, square_size, square_size))

    # green top
    pygame.draw.rect(tex, GREEN, (0, 0, BLOCK_WIDTH, 10))

    return tex

ground_texture = create_ground_texture()

def draw_ground_segment(segment):
    game_surface.blit(ground_texture, (segment.x, segment.y))

def generate_ground(hole_probability):
    global ground_segments, holes, streetlights, trees
    ground_segments = []
    holes = []
    streetlights.clear()
    trees.clear()

    x = 0
    for _ in range(SAFE_GROUND_LENGTH):
        seg = pygame.Rect(x, GAME_HEIGHT - GROUND_HEIGHT, BLOCK_WIDTH, GROUND_HEIGHT)
        ground_segments.append(seg)
        place_decor(seg)
        x += BLOCK_WIDTH

    while x < GAME_WIDTH:
        if random.random() < hole_probability:
            hole_width = random.randint(HOLE_WIDTH_MIN, HOLE_WIDTH_MAX)
            holes.append({"x": x, "width": hole_width, "passed": False})
            x += hole_width

        platform_size = random.randint(1, 2)
        for _ in range(platform_size):
            seg = pygame.Rect(x, GAME_HEIGHT - GROUND_HEIGHT, BLOCK_WIDTH, GROUND_HEIGHT)
            ground_segments.append(seg)
            place_decor(seg)
            x += BLOCK_WIDTH

def add_one_segment(hole_probability):
    if ground_segments:
        new_x = ground_segments[-1].x + BLOCK_WIDTH
    else:
        new_x = GAME_WIDTH

    if random.random() < hole_probability:
        hole_width = random.randint(HOLE_WIDTH_MIN, HOLE_WIDTH_MAX)
        holes.append({"x": new_x, "width": hole_width, "passed": False})
        new_x += hole_width

    seg = pygame.Rect(new_x, GAME_HEIGHT - GROUND_HEIGHT, BLOCK_WIDTH, GROUND_HEIGHT)
    ground_segments.append(seg)
    place_decor(seg)

# -----------------------------
# CARVE OUT HOLES AFTER DRAWING
# -----------------------------
def carve_out_holes():
    """
    Overwrite the hole area with HOLE_COLOR, ensuring a clean hole with no lines.
    """
    for h in holes:
        hole_rect = pygame.Rect(h["x"], GAME_HEIGHT - GROUND_HEIGHT, h["width"], GROUND_HEIGHT)
        pygame.draw.rect(game_surface, HOLE_COLOR, hole_rect)

# -----------------------------
# MAIN GAME
# -----------------------------
def main():
    global ai_prediction, confidence_score

    generate_sky()
    distance_travelled = 0.0
    scroll_speed, hole_prob = update_difficulty(distance_travelled)
    generate_ground(hole_prob)

    bird = Bird()
    clock = pygame.time.Clock()

    start_menu = True
    game_over = False
    game_started = False
    score = 0

    # Attempt camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Warning: Could not open webcam. AI will not function.")
        cap = None

    running = True
    while running:
        # 1) Fill sky with an 8-bit chunk-based gradient
        fill_sky_gradient()

        if start_menu:
            pygame.draw.rect(game_surface, BLACK, (0, 0, GAME_WIDTH, GAME_HEIGHT))
            title_text = menu_font.render("SCREAMING BIRD", True, WHITE)
            title_rect = title_text.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2 - 50))
            game_surface.blit(title_text, title_rect)

            sub_text = menu_font.render("PRESS SPACE TO START", True, WHITE)
            sub_rect = sub_text.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2 + 50))
            game_surface.blit(sub_text, sub_rect)

            scale_and_letterbox()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        start_menu = False
                        game_started = True
            clock.tick(30)
            continue

        if game_over:
            pygame.draw.rect(game_surface, BLACK, (0, 0, GAME_WIDTH, GAME_HEIGHT))
            over_text = menu_font.render("GAME OVER", True, RED)
            over_rect = over_text.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2 - 50))
            game_surface.blit(over_text, over_rect)

            score_text_surf = menu_font.render(f"FINAL SCORE: {score}", True, WHITE)
            score_rect = score_text_surf.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2 + 10))
            game_surface.blit(score_text_surf, score_rect)

            retry_text = menu_font.render("PRESS SPACE TO PLAY AGAIN", True, WHITE)
            retry_rect = retry_text.get_rect(center=(GAME_WIDTH//2, GAME_HEIGHT//2 + 70))
            game_surface.blit(retry_text, retry_rect)

            scale_and_letterbox()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        bird = Bird()
                        distance_travelled = 0
                        scroll_speed, hole_prob = update_difficulty(distance_travelled)
                        generate_ground(hole_prob)
                        score = 0
                        game_over = False
                        game_started = True
            clock.tick(30)
            continue

        # Normal game
        scroll_speed, hole_prob = update_difficulty(distance_travelled)
        update_clouds(scroll_speed)
        draw_clouds_and_dots()

        # Streetlights & trees
        update_streetlights(scroll_speed)
        draw_streetlights()
        update_trees(scroll_speed)
        draw_trees()

        # AI Inference
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                normalized_image_array = (resized.astype(np.float32) / 127.5) - 1.0
                data = np.expand_dims(normalized_image_array, axis=0)
                prediction = model.predict(data)
                index = np.argmax(prediction)
                ai_prediction = class_names[index]
                confidence_score = float(prediction[0][index])

        print(f"AI Prediction: '{ai_prediction}' (Confidence: {confidence_score:.2f})")

        if ai_prediction == "0 Jump" and confidence_score > 0.80:
            bird.jump()

        if game_started and not game_over:
            if not bird.update():
                game_over = True

            # Scroll ground
            for seg in ground_segments:
                seg.x -= scroll_speed

            # Scroll holes
            for h in holes:
                h["x"] -= scroll_speed
                # Score increment
                if h["x"] + h["width"] < bird.x and not h["passed"]:
                    h["passed"] = True
                    score += 1

            # Remove off-screen ground
            while ground_segments and ground_segments[0].x + BLOCK_WIDTH < 0:
                ground_segments.pop(0)
                add_one_segment(hole_prob)

            distance_travelled += scroll_speed

        # 2) Draw ground segments
        for seg in ground_segments:
            draw_ground_segment(seg)

        # 3) Overwrite holes with HOLE_COLOR to create a “clean hole”
        carve_out_holes()

        # 4) Draw bird
        bird.draw()

        # Score
        score_text_surf = score_font.render(f"Score: {score}", True, WHITE)
        game_surface.blit(score_text_surf, (10, 10))

        scale_and_letterbox()
        clock.tick(30)

    if cap is not None:
        cap.release()
    pygame.quit()

def scale_and_letterbox():
    scalex = SCREEN_WIDTH / GAME_WIDTH
    scaley = SCREEN_HEIGHT / GAME_HEIGHT
    scale = min(scalex, scaley)

    final_w = int(GAME_WIDTH * scale)
    final_h = int(GAME_HEIGHT * scale)

    scaled_surf = pygame.transform.scale(game_surface, (final_w, final_h))
    offset_x = (SCREEN_WIDTH - final_w) // 2
    offset_y = (SCREEN_HEIGHT - final_h) // 2

    screen.fill(BLACK)
    screen.blit(scaled_surf, (offset_x, offset_y))
    pygame.display.flip()

if __name__ == "__main__":
    main()
