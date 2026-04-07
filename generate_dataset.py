from PIL import Image, ImageDraw
import random
from pathlib import Path

background_dir = Path("backgrounds")
letters_dir = Path("arm_alph")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

backgrounds = list(background_dir.glob("*.*"))
letters = list(letters_dir.glob("*.*"))

if not backgrounds:
    raise ValueError("В папке 'backgrounds' нет файлов")
if not letters:
    raise ValueError("В папке 'arm_alph' нет файлов")

variants_per_letter = 5

min_fraction = 0.06   # 6–12% высоты фона — размер буквы
max_fraction = 0.12

min_angle = -45
max_angle = 45

# настройки «шумов»
light_leak_prob = 0.35      # вероятность добавить засвет
white_squares_prob = 0.35   # вероятность добавить белые квадраты

def add_light_leak(img: Image.Image) -> Image.Image:
    """Простейший light leak: полупрозрачный цветной градиент с края."""
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # случайный край
    side = random.choice(["left", "right", "top", "bottom"])

    # цвет засвета (оранжевый/красный, как плёночный)
    base_color = random.choice([
        (255, 200, 150),
        (255, 180, 120),
        (255, 160, 160),
        (255, 220, 180),
    ])

    # количество «полос» градиента
    steps = 10
    for i in range(steps):
        # от более яркого к более прозрачному
        alpha = int(255 * (1 - i / steps) * random.uniform(0.15, 0.4))
        color = (*base_color, alpha)

        if side == "left":
            x0 = 0
            x1 = int(w * 0.25 * (i + 1) / steps)
            box = (x0, 0, x1, h)
        elif side == "right":
            x1 = w
            x0 = w - int(w * 0.25 * (i + 1) / steps)
            box = (x0, 0, x1, h)
        elif side == "top":
            y0 = 0
            y1 = int(h * 0.25 * (i + 1) / steps)
            box = (0, y0, w, y1)
        else:  # bottom
            y1 = h
            y0 = h - int(h * 0.25 * (i + 1) / steps)
            box = (0, y0, w, y1)

        draw.rectangle(box, fill=color)

    return Image.alpha_composite(img, overlay)  # [web:89][web:21]

def add_white_squares(img: Image.Image) -> Image.Image:
    """Случайные белые квадраты/пропуски."""
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # сколько квадратов добавить
    n_squares = random.randint(3, 10)

    for _ in range(n_squares):
        # размер квадрата как доля картинки
        sq_size = random.randint(int(min(w, h) * 0.03), int(min(w, h) * 0.10))

        x0 = random.randint(0, max(1, w - sq_size))
        y0 = random.randint(0, max(1, h - sq_size))
        x1 = x0 + sq_size
        y1 = y0 + sq_size

        # иногда полностью белые, иногда полупрозрачные
        alpha = random.randint(120, 255)
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, alpha))  # [web:89][web:90]

    return Image.alpha_composite(img, overlay)

for letter_path in letters:
    letter_name = letter_path.stem

    for k in range(variants_per_letter):
        bg_path = random.choice(backgrounds)

        bg = Image.open(bg_path).convert("RGBA")
        letter = Image.open(letter_path).convert("RGBA")

        # размер буквы как доля высоты фона
        fraction = random.uniform(min_fraction, max_fraction)
        target_size = int(bg.height * fraction)

        stretch_to_square = random.random() < 0.5  # 50/50

        if stretch_to_square:
            # Растянуть в квадрат
            letter = letter.resize(
                (target_size, target_size),
                Image.Resampling.LANCZOS
            )
        else:
            # Сохранить пропорции
            scale = target_size / letter.height
            target_width = int(letter.width * scale)

            letter = letter.resize(
                (target_width, target_size),
                Image.Resampling.LANCZOS
            )

        # случайный поворот
        angle = random.uniform(min_angle, max_angle)
        letter = letter.rotate(angle,
                               resample=Image.Resampling.BICUBIC,
                               expand=True)  # [web:13][web:78]

        # подстраховка, чтобы влезало
        if letter.width >= bg.width or letter.height >= bg.height:
            ratio = min((bg.width * 0.9) / letter.width,
                        (bg.height * 0.9) / letter.height)
            letter = letter.resize(
                (int(letter.width * ratio), int(letter.height * ratio)),
                Image.Resampling.LANCZOS
            )

        max_x = bg.width - letter.width
        max_y = bg.height - letter.height
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        layer.paste(letter, (x, y), letter)
        result = Image.alpha_composite(bg, layer)  # [web:21]

        # накладываем эффекты «кривой камеры»
        if random.random() < light_leak_prob:
            result = add_light_leak(result)        # [web:89]
        if random.random() < white_squares_prob:
            result = add_white_squares(result)     # [web:89][web:90]

        out_name = f"{letter_name}_{k:03}.png"
        result.save(output_dir / out_name)

print("all done!")