"""
纹理生成工具
"""

import colorsys
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from .logging_utils import setup_logging


class TextureGenerator:
    """
    生成多种合成纹理，含网格编号与对比度优化。
    """

    def __init__(self, image_size=(1024, 1024), num_squares=16):
        self.image_size = image_size
        self.num_squares = num_squares
        self.square_size = image_size[0] // num_squares

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    def _load_font(self, font_size: int):
        font_paths = [
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
                logger.info("Successfully loaded font: %s", font_path)
                return font
            except IOError:
                continue
        logger.warning("Could not load any TrueType font, using default font")
        return ImageFont.load_default()

    def generate_unique_color_texture(self, filepath: str, seed: int, font_scale: float = 0.7):
        # 使用局部随机数生成器确保多进程安全
        rng = random.Random(seed)
        image = Image.new('RGB', self.image_size)
        draw = ImageDraw.Draw(image)

        colors = []
        for _ in range(self.num_squares * self.num_squares):
            hue = rng.random()
            saturation = rng.uniform(0.6, 0.9)
            value = rng.uniform(0.5, 0.9)
            color = self._hsv_to_rgb(hue, saturation, value)
            colors.append(color)

        font_size = int(self.square_size * font_scale)
        logger.info("Font size: %d, Square size: %d", font_size, self.square_size)
        font = self._load_font(font_size)

        for i in range(self.num_squares):
            for j in range(self.num_squares):
                color_idx = i * self.num_squares + j
                x0, y0 = i * self.square_size, j * self.square_size
                x1, y1 = x0 + self.square_size, y0 + self.square_size
                draw.rectangle([x0, y0, x1, y1], fill=colors[color_idx])

                text_content = str(color_idx)
                text_bbox = draw.textbbox((0, 0), text_content, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x0 + (self.square_size - text_width) / 2
                text_y = y0 + (self.square_size - text_height) / 2

                bg_color = colors[color_idx]
                brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                draw.text((text_x, text_y), text_content, fill=text_color, font=font)

        image.save(filepath)
        logger.info("Generated unique color texture and saved to %s", filepath)
        return image

    def generate_checkerboard_texture(self, filepath: str, seed: int, font_scale: float = 0.7):
        # 使用局部随机数生成器确保多进程安全
        rng = random.Random(seed)
        image = Image.new('RGB', self.image_size)
        draw = ImageDraw.Draw(image)

        font_size = int(self.square_size * font_scale)
        font = self._load_font(font_size)

        for i in range(self.num_squares):
            for j in range(self.num_squares):
                color_idx = (i + j) % 2
                x0, y0 = i * self.square_size, j * self.square_size
                x1, y1 = x0 + self.square_size, y0 + self.square_size
                color = (255, 255, 255) if color_idx == 0 else (0, 0, 0)
                draw.rectangle([x0, y0, x1, y1], fill=color)

                text_content = str(i * self.num_squares + j)
                text_bbox = draw.textbbox((0, 0), text_content, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x0 + (self.square_size - text_width) / 2
                text_y = y0 + (self.square_size - text_height) / 2
                text_color = (0, 0, 0) if color_idx == 0 else (255, 255, 255)
                draw.text((text_x, text_y), text_content, fill=text_color, font=font)

        image.save(filepath)
        logger.info("Generated checkerboard texture and saved to %s", filepath)
        return image

    def generate_gradient_texture(self, filepath: str, seed: int, font_scale: float = 0.7):
        # 使用局部随机数生成器确保多进程安全
        rng = random.Random(seed)
        image = Image.new('RGB', self.image_size)
        draw = ImageDraw.Draw(image)

        font_size = int(self.square_size * font_scale)
        font = self._load_font(font_size)

        for i in range(self.num_squares):
            for j in range(self.num_squares):
                x0, y0 = i * self.square_size, j * self.square_size
                x1, y1 = x0 + self.square_size, y0 + self.square_size
                hue = (i + j) / (2 * self.num_squares)
                saturation = 0.8
                value = 0.7 + 0.2 * (i / self.num_squares)
                color = self._hsv_to_rgb(hue, saturation, value)
                draw.rectangle([x0, y0, x1, y1], fill=color)

                text_content = str(i * self.num_squares + j)
                text_bbox = draw.textbbox((0, 0), text_content, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x0 + (self.square_size - text_width) / 2
                text_y = y0 + (self.square_size - text_height) / 2
                draw.text((text_x, text_y), text_content, fill=(255, 255, 255), font=font)

        image.save(filepath)
        logger.info("Generated gradient texture and saved to %s", filepath)
        return image


__all__ = ["TextureGenerator"]


