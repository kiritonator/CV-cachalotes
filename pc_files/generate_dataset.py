from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import random
from pathlib import Path

setting = input("train/val/test: ")
number = int(input("number: "))
background_dir = Path("real_backgrounds")
letters_dir = Path("real_arm_alph")
output_dir_im = Path(f"real_dataset3/{setting}/images")
output_dir_txt = Path(f"real_dataset3/{setting}/labels")

backgrounds = list(background_dir.glob("*.*"))
letters = list(letters_dir.glob("*.*"))

if not backgrounds:
    raise ValueError("В папке 'backgrounds' нет файлов")
if not letters:
    raise ValueError("В папке 'arm_alph' нет файлов")

variants_per_letter = number

min_fraction = 0.06   # % высоты фона — размер буквы
max_fraction = 0.15

min_angle = -360
max_angle = 360

# настройки «шумов»
light_leak_prob = 0.4      # вероятность добавить засвет
white_squares_prob = 0.26   # вероятность добавить белые квадраты

def add_light_leak(img: Image.Image) -> Image.Image:
    """градиент с краев в качестве засвета"""
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    side = random.choice(["left", "right", "top", "bottom"])

    base_color = random.choice([
        (255, 200, 150),
        (255, 180, 120),
        (255, 160, 160),
        (255, 220, 180),
    ])

    steps = 10
    for i in range(steps):
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
        else:
            y1 = h
            y0 = h - int(h * 0.25 * (i + 1) / steps)
            box = (0, y0, w, y1)

        draw.rectangle(box, fill=color)

    return Image.alpha_composite(img, overlay)

def add_white_squares(img: Image.Image) -> Image.Image:
    """Случайные белые квадраты/пропуски."""
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    n_squares = random.randint(3, 10)

    for _ in range(n_squares):
        sq_size = random.randint(int(min(w, h) * 0.03), int(min(w, h) * 0.10))

        x0 = random.randint(0, max(1, w - sq_size))
        y0 = random.randint(0, max(1, h - sq_size))
        x1 = x0 + sq_size
        y1 = y0 + sq_size

        alpha = random.randint(120, 255)
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, alpha))

    return Image.alpha_composite(img, overlay)

def add_purple_pink_tint(img: Image.Image) -> Image.Image:
    tint_color = random.choice([
        (255, 235, 248),  # едва розоватый
        (248, 235, 255),  # едва лиловый
        (242, 232, 255),  # холодный фиолетовый
        (255, 230, 242),  # тёплый розовый
        (235, 225, 255),  # слабый пурпурно-синий
        (250, 235, 255),  # светлая фуксия
        (230, 180, 208),  # приглушённый розовый
        (225, 175, 225),  # серо-лиловый
        (210, 175, 245),  # холодный фиолетовый
        (200, 165, 235)  # приглушённый пурпурный
    ])
    alpha = random.randint(8, 30)

    overlay = Image.new(
        "RGBA",
        img.size,
        (*tint_color, alpha),
    )

    return Image.alpha_composite(img, overlay)

def light_letter(letter: Image.Image) -> Image.Image:
    factor = random.uniform(0.3, 1.7)
    return ImageEnhance.Brightness(letter).enhance(factor)

def deform_letter(letter: Image.Image) -> Image.Image:
    w, h = letter.size

    scale_x = random.uniform(0.92, 1.08)
    scale_y = random.uniform(0.92, 1.08)

    new_w = max(1, int(w * scale_x))
    new_h = max(1, int(h * scale_y))

    letter = letter.resize(
        (new_w, new_h),
        Image.Resampling.BICUBIC,
    )

    left = random.randint(0, min(3, max(0, new_w // 10)))
    right = random.randint(0, min(3, max(0, new_w // 10)))
    top = random.randint(0, min(3, max(0, new_h // 10)))
    bottom = random.randint(0, min(3, max(0, new_h // 10)))

    if new_w - left - right > 2 and new_h - top - bottom > 2:
        letter = letter.crop((
            left,
            top,
            new_w - right,
            new_h - bottom,
        ))

    return letter

def blur_letter_edges(letter: Image.Image) -> Image.Image:
    radius = random.uniform(0.3, 1.2)

    alpha = letter.getchannel("A")
    blurred_alpha = alpha.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )

    result = letter.copy()
    result.putalpha(blurred_alpha)

    return result

for letter_path in letters:
    letter_name = letter_path.stem

    for k in range(variants_per_letter):
        bg_path = random.choice(backgrounds)

        bg = Image.open(bg_path).convert("RGBA")
        letter = Image.open(letter_path).convert("RGBA")

        fraction = random.uniform(min_fraction, max_fraction)
        target_size = int(bg.height * fraction)

        stretch_to_square = random.random() < 0.5

        if stretch_to_square:
            letter = letter.resize(
                (target_size, target_size),
                Image.Resampling.LANCZOS
            )
        else:
            scale = target_size / letter.height
            target_width = int(letter.width * scale)

            letter = letter.resize(
                (target_width, target_size),
                Image.Resampling.LANCZOS
            )
        if random.random() < 0.5:
            letter = deform_letter(letter)

        angle = random.uniform(min_angle, max_angle)
        letter = letter.rotate(angle,
                               resample=Image.Resampling.BICUBIC,
                               expand=True)

        if random.random() < 0.4:
            letter = light_letter(letter)

        if random.random() < 0.4:
            letter = add_purple_pink_tint(letter)

        if random.random() < 0.6:
            letter = blur_letter_edges(letter)

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
        result = Image.alpha_composite(bg, layer)

        if random.random() < light_leak_prob:
            result = add_light_leak(result)
        if random.random() < white_squares_prob:
            result = add_white_squares(result)

        out_name = f"{letter_name}_{k:03}.png"
        result.save(output_dir_im / out_name)


        file_name = f"{letter_name}_{k:03}.txt"
        file_path = output_dir_txt / file_name
        x_center = (x + letter.width / 2) / bg.width
        y_center = (y + letter.height / 2) / bg.height



        with file_path.open("w", encoding="utf-8") as f:
            f.write(
                f"{letter_name[:-1]} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{letter.width / bg.width:.6f} {letter.height / bg.height:.6f}\n"
            )

print("all done!")