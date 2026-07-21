"""
Generate STT application icons in all required formats.

Outputs:
    icon.png          — 512x512 source (Linux / general)
    icon.ico          — multi-resolution Windows icon
    icon.icns         — macOS icon bundle (requires macOS + iconutil)

Run before PyInstaller:
    python make_icon.py
"""

import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw


# ── Design constants ──────────────────────────────────────────────────────────
# Pillow port of packaging/icon-source.svg (256×256 viewBox); coordinates below
# are in that viewBox space, scaled by `u`. Keep the two files in sync.

BG_COLOR     = (243, 242, 242)  # light  #f3f2f2
FG_COLOR     = (32, 30, 29)     # dark   #201e1d
ACCENT_COLOR = (236, 48, 19)    # red    #ec3013

SUPERSAMPLE = 4  # render N× then LANCZOS-downscale so 16-32px sizes stay crisp


def draw_icon(size: int) -> Image.Image:
    s    = size * SUPERSAMPLE
    u    = s / 256                # SVG viewBox unit
    img  = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lw   = 12 * u                 # SVG stroke-width

    # Rounded-square background plate (the SVG's square bg, platform-styled)
    pad = s / 16
    draw.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=s / 5,
        fill=BG_COLOR,
    )

    # ── Microphone body (rounded-rect outline) ───────────────────────────────
    draw.rounded_rectangle(
        [90 * u, 34 * u, 166 * u, 156 * u],
        radius=38 * u,
        outline=FG_COLOR,
        width=round(lw),
    )

    # ── Level bars ───────────────────────────────────────────────────────────
    for x, y, w, h in [(106, 80, 9, 30), (123, 66, 9, 58), (140, 80, 9, 30)]:
        draw.rectangle([x * u, y * u, (x + w) * u, (y + h) * u], fill=ACCENT_COLOR)

    # ── Stand: two verticals + bottom semicircle (stroke centred on path) ────
    # Verticals at x=72 and x=184, y 132→142, square caps extending 6 up.
    for x in (72, 184):
        draw.rectangle(
            [(x - 6) * u, (132 - 6) * u, (x + 6) * u, 142 * u],
            fill=FG_COLOR,
        )
    # Semicircle: centreline radius 56 around (128, 142); bbox is outer edge.
    draw.arc(
        [(128 - 62) * u, (142 - 62) * u, (128 + 62) * u, (142 + 62) * u],
        start=0, end=180,
        fill=FG_COLOR,
        width=round(lw),
    )

    # ── Stem and base ────────────────────────────────────────────────────────
    draw.rectangle([121 * u, 198 * u, 135 * u, 220 * u], fill=FG_COLOR)
    draw.rectangle([94 * u, 216 * u, 162 * u, 229 * u], fill=FG_COLOR)

    return img.resize((size, size), Image.LANCZOS)


# ── Export helpers ────────────────────────────────────────────────────────────

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]

ICNS_SIZES = {
    "icon_16x16.png":      16,
    "icon_16x16@2x.png":   32,
    "icon_32x32.png":      32,
    "icon_32x32@2x.png":   64,
    "icon_128x128.png":    128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png":    256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png":    512,
    "icon_512x512@2x.png": 1024,
}


def make_ico(path: str = "icon.ico"):
    images = [draw_icon(s).convert("RGBA") for s in ICO_SIZES]
    # Save from the LARGEST frame: Pillow silently drops any requested size
    # bigger than the base image, so saving from 16px yields a 16px-only .ico.
    images[-1].save(
        path, format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=images[:-1],
    )
    print(f"  {path}")


def make_png(path: str = "icon.png", size: int = 512):
    draw_icon(size).save(path, format="PNG")
    print(f"  {path}")


def make_icns(path: str = "icon.icns"):
    iconset = "icon.iconset"
    os.makedirs(iconset, exist_ok=True)
    for name, sz in ICNS_SIZES.items():
        draw_icon(sz).save(os.path.join(iconset, name), format="PNG")
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", iconset, "-o", path],
            check=True, capture_output=True,
        )
        print(f"  {path}")
    except FileNotFoundError:
        print("  icon.icns skipped (iconutil not available — macOS only)")
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating icons...")
    make_ico()
    make_png()
    if sys.platform == "darwin":
        make_icns()
    print("Done.")
