"""
Generate favicon assets from the real orracle logo.

Run from the orracle project root:
    python3 branding/generate_favicons.py

Outputs to static/:
    favicon.png         — 32x32
    favicon-16.png / -32.png / -48.png
    apple-touch-icon.png — 192x192
    favicon.ico         — 16/32/48 multi-size bundle

The orracle glyph is a gold/amber eye with a copper flame inside a cream
halo. Unlike orrapus/concord which are G-dominant greens, orracle is
Y-dominant (R and G both high, B low). A raw LANCZOS downscale washes
out the warm tones against the cream halo at tab size, so we:

    1. Detect gold-family pixels via R > B + 30 AND |R - G| < 100
       — catches gold (#F8E898), amber, and bronze flame (#884808)
       without grabbing the pink or green canvas bleed or the cream
       halo itself.

    2. Clamp those pixels to α=255 and blend 40/60 toward Oracle Gold
       #F8E898 so the color density survives downscaling.

    3. Tight-crop to the alpha bbox and pad to square with transparent,
       then LANCZOS downscale with per-size unsharp-mask sharpening.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "branding" / "logo.png"
STATIC = ROOT / "static"

# Oracle Gold — the contrast-boost target
BRAND_PRIMARY = np.array([248, 232, 152], dtype=np.float32)


def build_clean_glyph(src: Image.Image) -> Image.Image:
    """Isolate gold-family pixels and boost them toward brand gold.

    Multi-gate filter:
      1. Warm bias — R > B (gold is warm)
      2. Saturation — max(RGB) - min(RGB) > 50 (rejects the cream halo
         which has near-white R=G=B and would otherwise pass the warm
         test because cream is technically slightly warm)
      3. Yellow-ish hue — |R - G| < 100 (rejects pink canvas where
         R >> G, rejects red)
      4. Reasonable opacity — α > 100 (rejects faint canvas bleed)
    """
    arr = np.array(src).astype(np.float32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)
    saturation = rgb_max - rgb_min

    is_glyph = (
        (r > b)
        & (saturation > 50)
        & (np.abs(r - g) < 100)
        & (a > 100)
    )

    new_alpha = np.zeros_like(a)
    new_alpha[is_glyph] = 255.0

    out_rgb = arr[..., :3].copy()
    blend = is_glyph[..., None]
    out_rgb = np.where(blend, out_rgb * 0.4 + BRAND_PRIMARY * 0.6, out_rgb)

    stacked = np.dstack([out_rgb, new_alpha]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(stacked, "RGBA")


def tight_crop_square(img: Image.Image) -> Image.Image:
    alpha = np.array(img)[..., 3]
    mask = alpha > 20
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = int(max(x1 - x0, y1 - y0) * 0.05)
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(img.width, x1 + margin)
    y1 = min(img.height, y1 + margin)

    cropped = img.crop((x0, y0, x1, y1))
    cw, ch = cropped.size
    side = max(cw, ch)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(cropped, ((side - cw) // 2, (side - ch) // 2), cropped)
    return square


def scale_to(square: Image.Image, sz: int) -> Image.Image:
    im = square.resize((sz, sz), Image.LANCZOS)
    if sz <= 16:
        im = im.filter(ImageFilter.UnsharpMask(radius=0.5, percent=220, threshold=1))
    elif sz <= 32:
        im = im.filter(ImageFilter.UnsharpMask(radius=0.6, percent=170, threshold=2))
    elif sz <= 48:
        im = im.filter(ImageFilter.UnsharpMask(radius=0.7, percent=150, threshold=2))
    else:
        im = im.filter(ImageFilter.UnsharpMask(radius=0.8, percent=130, threshold=2))
    return im


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    src = Image.open(SRC).convert("RGBA")
    print(f"source: {SRC} {src.size}")

    clean = build_clean_glyph(src)
    square = tight_crop_square(clean)
    print(f"squared transparent: {square.size}")

    sizes = {
        16: "favicon-16.png",
        32: "favicon-32.png",
        48: "favicon-48.png",
        192: "apple-touch-icon.png",
    }
    for sz, name in sizes.items():
        scale_to(square, sz).save(STATIC / name, "PNG", optimize=True)
        print(f"  static/{name}")

    (STATIC / "favicon.png").write_bytes((STATIC / "favicon-32.png").read_bytes())
    scale_to(square, 48).save(
        STATIC / "favicon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48)],
    )
    print("  static/favicon.ico (16/32/48)")

    # Keep the in-page sidebar image as the full-resolution original
    src.save(STATIC / "orracle_logo.png", "PNG", optimize=True)
    print("  static/orracle_logo.png")


if __name__ == "__main__":
    main()
