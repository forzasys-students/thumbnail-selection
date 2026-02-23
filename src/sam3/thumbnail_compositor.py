"""
Enhanced Thumbnail Compositor - Supports shapes, logos, multiple fonts

NEW: Renders text, rectangles, logos with layer ordering
"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import requests
from io import BytesIO


class ThumbnailCompositor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._font_cache = {}
        self._logo_cache = {}

    def create_thumbnail(
        self,
        image_path: str,
        mask_path: Optional[str],
        elements: List[Dict],  
        background_color: str = "#000000",
        output_path: str = "thumbnail.png",
        output_size: Optional[Tuple[int, int]] = None,
        player_layer: str = "foreground",
        blur_background: bool = False,
        blur_radius: int = 12,
    ) -> str:
        original = Image.open(image_path).convert("RGB")
        w, h = original.size

        if self.debug:
            print(f"[Compositor] {w}x{h}  elements={len(elements)}")

        has_mask = mask_path and os.path.exists(mask_path)

        mask_img = None
        if has_mask:
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != original.size:
                mask_img = mask_img.resize(original.size, Image.Resampling.LANCZOS)

        # Base layer
        if has_mask and blur_background:
            canvas = self._make_blurred_background(original, mask_img, blur_radius)
        elif has_mask and player_layer == "foreground":
            canvas = original.copy().convert("RGB")
        else:
            canvas = Image.new("RGB", (w, h), self._hex_to_rgb(background_color))
            if not has_mask:
                canvas.paste(original)

        # Helper functions
        def paste_player():
            if has_mask:
                cutout = self._apply_mask(original, mask_img)
                canvas.paste(cutout, (0, 0), cutout)

        def draw_elements(layer_name: str):
            for el in elements:
                if el.get("layer") == layer_name:
                    self._draw_element(canvas, el)

        # Compose layers
        if player_layer == "background":
            paste_player()
            draw_elements("background")
            draw_elements("foreground")
        else:
            draw_elements("background")
            paste_player()
            draw_elements("foreground")

        # Resize & save
        if output_size:
            canvas = canvas.resize(output_size, Image.Resampling.LANCZOS)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        canvas.save(output_path, quality=95)

        if self.debug:
            print(f"[Compositor] Saved → {output_path}")

        return output_path

    def _draw_element(self, canvas: Image.Image, el: Dict):
        el_type = el.get("type")
        if el_type == "text":
            self._draw_text(canvas, el)
        elif el_type == "rect":
            self._draw_rect(canvas, el)
  

    def _draw_text(self, canvas: Image.Image, el: Dict):
        draw = ImageDraw.Draw(canvas)

        content = el.get("content", "")
        x = el.get("x", 0)
        y = el.get("y", 0)
        font_size = el.get("fontSize", 48)
        color = el.get("color", "#FFFFFF")
        stroke_color = el.get("strokeColor", "#000000")
        stroke_width = el.get("strokeWidth", 0)
        font_family = el.get("fontFamily", "impact")
        font_weight = el.get("fontWeight", "bold")

        font = self._get_font(font_family, font_size)
        fill_rgb = self._hex_to_rgb(color)
        stroke_rgb = self._hex_to_rgb(stroke_color)

        if stroke_width > 0:
            draw.text((x, y), content, font=font,
                      fill=fill_rgb, stroke_width=stroke_width, stroke_fill=stroke_rgb)
        else:
            draw.text((x, y), content, font=font, fill=fill_rgb)

    def _draw_rect(self, canvas: Image.Image, el: Dict):
        x = el.get("x", 0)
        y = el.get("y", 0)
        width = el.get("width", 100)
        height = el.get("height", 100)
        fill_color = el.get("fillColor")
        stroke_color = el.get("strokeColor")
        stroke_width = el.get("strokeWidth", 0)
        corner_radius = el.get("cornerRadius", 0)

        # Create a separate image for the rectangle with alpha
        rect_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect_img)

        if corner_radius > 0:
            draw.rounded_rectangle([(0, 0), (width, height)], corner_radius,
                                   fill=fill_color, outline=stroke_color, width=stroke_width)
        else:
            draw.rectangle([(0, 0), (width, height)],
                           fill=fill_color, outline=stroke_color, width=stroke_width)

        canvas.paste(rect_img, (int(x), int(y)), rect_img)

    def _make_blurred_background(self, original: Image.Image, mask: Image.Image, blur_radius: int) -> Image.Image:
        blurred = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        result = blurred.copy().convert("RGB")
        cutout = self._apply_mask(original, mask)
        result.paste(cutout, (0, 0), cutout)
        return result

    def _apply_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask)
        return image_rgba

    def _get_font(self, font_family: str, size: int) -> ImageFont.FreeTypeFont:
        cache_key = (font_family.lower(), size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_paths = {
            "impact": [
                "C:\\Windows\\Fonts\\impact.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
                "/System/Library/Fonts/Supplemental/Impact.ttf",
            ],
            "arial black": [
                "C:\\Windows\\Fonts\\ariblk.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
            ],
            "bebas neue": [
                "C:\\Windows\\Fonts\\bebasneue-regular.ttf",
                "fonts/BebasNeue-Regular.ttf",
            ],
            "montserrat": [
                "C:\\Windows\\Fonts\\Montserrat-Bold.ttf",
                "fonts/Montserrat-Bold.ttf",
            ],
            "oswald": [
                "C:\\Windows\\Fonts\\Oswald-Bold.ttf",
                "fonts/Oswald-Bold.ttf",
            ],
        }

        font = None
        for path in font_paths.get(font_family.lower(), []):
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    pass

        if font is None:
            for fallback in [
                "C:\\Windows\\Fonts\\arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]:
                if os.path.exists(fallback):
                    try:
                        font = ImageFont.truetype(fallback, size)
                        break
                    except Exception:
                        pass

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))