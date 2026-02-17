"""
Thumbnail Compositor - Backend Image Composition

"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2


class ThumbnailCompositor:
    """
    Layer order when player_layer == 'foreground' (default):
        1. Background colour (or blurred image)
        2. Background text
        3. Player cutout
        4. Foreground text

    Layer order when player_layer == 'background':
        1. Background colour (or blurred image)
        2. Player cutout
        3. Background text
        4. Foreground text
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._font_cache = {}

    # -------------------------------------------------------------------------
    def create_thumbnail(
        self,
        image_path: str,
        mask_path: Optional[str],
        text_elements: List[Dict],
        background_color: str = "#000000",
        output_path: str = "thumbnail.png",
        output_size: Optional[Tuple[int, int]] = None,
        # ── NEW PARAMS ──────────────────────────────────────────────────────
        player_layer: str = "foreground",   # 'foreground' | 'background'
        blur_background: bool = False,       # blur everything except player
        blur_radius: int = 12,              # px radius for background blur
    ) -> str:
        original = Image.open(image_path).convert("RGB")
        w, h = original.size

        if self.debug:
            print(f"[Compositor] {w}x{h}  player_layer={player_layer}  blur_bg={blur_background}")

        has_mask = mask_path and os.path.exists(mask_path)

        # ── Load mask ────────────────────────────────────────────────────────
        mask_img = None
        if has_mask:
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != original.size:
                mask_img = mask_img.resize(original.size, Image.Resampling.LANCZOS)

        # ── Base layer ───────────────────────────────────────────────────────
        if has_mask and blur_background:
            # Blurred photo; sharp player will be pasted on top later
            canvas = self._make_blurred_background(original, mask_img, blur_radius)
        elif has_mask and player_layer == "foreground":
            # Foreground mode: show full photo as background; player cutout is
            # pasted again on top of text so it visually pops out.
            canvas = original.copy().convert("RGB")
        else:
            # Background mode or no mask: solid colour background
            canvas = Image.new("RGB", (w, h), self._hex_to_rgb(background_color))
            if not has_mask:
                canvas.paste(original)

        # ── Player cutout helper ─────────────────────────────────────────────
        def paste_player():
            if has_mask:
                cutout = self._apply_mask(original, mask_img)
                canvas.paste(cutout, (0, 0), cutout)

        # ── Text helper ──────────────────────────────────────────────────────
        def draw_texts(layer_name: str):
            for t in text_elements:
                if t.get("layer") == layer_name:
                    self._draw_text(canvas, t)

        # ── Compose layers ───────────────────────────────────────────────────
        if player_layer == "background":
            # player → bg-text → fg-text
            paste_player()
            draw_texts("background")
            draw_texts("foreground")
        else:
            # bg-text → player → fg-text  (default / 'foreground')
            draw_texts("background")
            paste_player()
            draw_texts("foreground")

        # ── Resize & save ────────────────────────────────────────────────────
        if output_size:
            canvas = canvas.resize(output_size, Image.Resampling.LANCZOS)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        canvas.save(output_path, quality=95)

        if self.debug:
            print(f"[Compositor] Saved → {output_path}")

        return output_path

    # -------------------------------------------------------------------------
    def _make_blurred_background(
        self,
        original: Image.Image,
        mask: Image.Image,
        blur_radius: int,
    ) -> Image.Image:
        """
        Return an RGB image where:
        - the background is Gaussian-blurred
        - the masked player region is composited back at full sharpness
        """
        # Blur the whole image
        blurred = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        result  = blurred.copy().convert("RGB")

        # Paste the sharp player on top
        cutout = self._apply_mask(original, mask)  # RGBA
        result.paste(cutout, (0, 0), cutout)

        return result

    # -------------------------------------------------------------------------
    def _apply_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask)
        return image_rgba

    # -------------------------------------------------------------------------
    def _draw_text(self, canvas: Image.Image, text_element: Dict):
        draw = ImageDraw.Draw(canvas)

        content      = text_element.get("content", "")
        x            = text_element.get("x", 0)
        y            = text_element.get("y", 0)
        font_size    = text_element.get("fontSize", 48)
        color        = text_element.get("color", "#FFFFFF")
        stroke_color = text_element.get("strokeColor", "#000000")
        stroke_width = text_element.get("strokeWidth", 0)
        font_family  = text_element.get("fontFamily", "impact")

        font      = self._get_font(font_family, font_size)
        fill_rgb  = self._hex_to_rgb(color)
        stroke_rgb = self._hex_to_rgb(stroke_color)

        if stroke_width > 0:
            draw.text((x, y), content, font=font,
                      fill=fill_rgb, stroke_width=stroke_width, stroke_fill=stroke_rgb)
        else:
            draw.text((x, y), content, font=font, fill=fill_rgb)

        if self.debug:
            print(f"[Compositor] Text '{content}' @ ({x},{y}) layer={text_element.get('layer')}")

    # -------------------------------------------------------------------------
    def _get_font(self, font_family: str, size: int) -> ImageFont.FreeTypeFont:
        cache_key = (font_family.lower(), size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_paths = {
            "impact": [
                "C:\\Windows\\Fonts\\impact.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
                "/System/Library/Fonts/Supplemental/Impact.ttf",
                "fonts/Impact.ttf",
            ],
            "arial-black": [
                "C:\\Windows\\Fonts\\ariblk.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
                "/System/Library/Fonts/Supplemental/Arial Black.ttf",
                "fonts/Arial-Black.ttf",
            ],
            "bebas": [
                "fonts/BebasNeue-Regular.ttf",
                "/usr/share/fonts/truetype/bebas/BebasNeue-Regular.ttf",
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
                "/System/Library/Fonts/Helvetica.ttc",
            ]:
                if os.path.exists(fallback):
                    try:
                        font = ImageFont.truetype(fallback, size)
                        break
                    except Exception:
                        pass

        if font is None:
            font = ImageFont.load_default()
            if self.debug:
                print("[Compositor] Using default bitmap font")

        self._font_cache[cache_key] = font
        return font

    # -------------------------------------------------------------------------
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Thumbnail Compositor")
    parser.add_argument("--image",    required=True)
    parser.add_argument("--mask",     default=None)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--text",     default=None)
    parser.add_argument("--bg-color", default="#000000")
    parser.add_argument("--player-layer", default="foreground", choices=["foreground", "background"])
    parser.add_argument("--blur-background", action="store_true")
    parser.add_argument("--blur-radius", type=int, default=12)
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    text_elements = []
    if args.text:
        text_elements = json.load(open(args.text)) if os.path.exists(args.text) else json.loads(args.text)

    compositor = ThumbnailCompositor(debug=args.debug)
    out = compositor.create_thumbnail(
        image_path=args.image,
        mask_path=args.mask,
        text_elements=text_elements,
        background_color=args.bg_color,
        output_path=args.output,
        player_layer=args.player_layer,
        blur_background=args.blur_background,
        blur_radius=args.blur_radius,
    )
    print(f"[INFO] Thumbnail created: {out}")