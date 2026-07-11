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

BG_COLOR   = (15, 118, 110)   # teal
FG_COLOR   = (255, 255, 255)  # white


def draw_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s    = size

    # Rounded-square background
    pad = max(1, s // 16)
    draw.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=s // 5,
        fill=BG_COLOR,
    )

    # ── Microphone body (pill shape) ──────────────────────────────────────────
    bw = s * 0.28          # body width
    bh = s * 0.36          # body height
    cx = s / 2
    by = s * 0.37          # body vertical centre
    draw.rounded_rectangle(
        [cx - bw / 2, by - bh / 2, cx + bw / 2, by + bh / 2],
        radius=bw / 2,
        fill=FG_COLOR,
    )

    # ── Stand arc (U-shape below body) ───────────────────────────────────────
    ar   = s * 0.24        # arc radius
    arc_cy = s * 0.555     # centre of the arc's bounding circle
    lw   = max(2, s // 22)
    draw.arc(
        [cx - ar, arc_cy - ar, cx + ar, arc_cy + ar],
        start=0, end=180,
        fill=FG_COLOR,
        width=lw,
    )

    # ── Vertical stem ────────────────────────────────────────────────────────
    stem_top = arc_cy
    stem_bot = s * 0.79
    hw = lw / 2
    draw.rectangle(
        [cx - hw, stem_top, cx + hw, stem_bot],
        fill=FG_COLOR,
    )

    # ── Horizontal base ───────────────────────────────────────────────────────
    base_w = s * 0.28
    base_h = lw
    draw.rectangle(
        [cx - base_w / 2, stem_bot - base_h / 2,
         cx + base_w / 2, stem_bot + base_h / 2],
        fill=FG_COLOR,
    )

    return img


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
    images[0].save(
        path, format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
        append_images=images[1:],
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
