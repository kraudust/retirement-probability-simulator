"""Generate the app icon: assets/icon.icns (macOS) and assets/icon.ico (Windows).

    python3 tools/make_icon.py

The icon is generated rather than checked in as an opaque binary so it can be
adjusted later without hunting for whatever tool made it. The .icns step shells
out to `iconutil`, which is macOS-only; on other platforms only the .ico is
written (and the committed .icns is left alone).

Design: a rounded square in a muted green with a heavy white dollar sign and
nothing else -- readable at 16x16 in the Dock, which is the size
that actually matters. Anything more detailed turns to mush at that scale.
"""

import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

BG_TOP = (34, 110, 74)        # deep green
BG_BOTTOM = (22, 78, 54)      # slightly darker, for a vertical gradient
GLYPH = (255, 255, 255)

# Ordered largest-first so the biggest render is the reference for the rest.
ICNS_SIZES = [1024, 512, 256, 128, 64, 32, 16]
ICO_SIZES = [256, 128, 64, 48, 32, 16]

BOLD_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def _font(px):
    """Heaviest available system font at `px`, or PIL's bitmap default."""
    for path in BOLD_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                continue
    return ImageFont.load_default()


def render(size):
    """One square RGBA icon at `size` pixels."""
    # Draw at 4x and downsample: PIL has no antialiased shape drawing, so
    # supersampling is what keeps the rounded corners and glyph edges clean.
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    # vertical gradient inside a rounded-rect mask
    grad = Image.new("RGBA", (s, s))
    gd = ImageDraw.Draw(grad)
    for y in range(s):
        t = y / max(1, s - 1)
        gd.line([(0, y), (s, y)],
                fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=255)
    img.paste(grad, (0, 0), mask)

    # Deliberately nothing behind the glyph. A chart line crossing the "$" turned
    # into visual noise at Dock size -- an icon has to survive being 16px wide,
    # and at that scale one bold shape beats two competing ones.

    # the dollar sign, optically centred (font metrics rarely centre it for you)
    font = _font(int(s * 0.72))
    box = draw.textbbox((0, 0), "$", font=font)
    draw.text(((s - (box[2] - box[0])) / 2 - box[0],
               (s - (box[3] - box[1])) / 2 - box[1]),
              "$", font=font, fill=GLYPH)

    return img.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    renders = {n: render(n) for n in ICNS_SIZES}

    renders[1024].save(os.path.join(ASSETS, "icon.png"))
    renders[256].save(os.path.join(ASSETS, "icon.ico"),
                      sizes=[(n, n) for n in ICO_SIZES])
    print(f"  wrote assets/icon.png and assets/icon.ico ({len(ICO_SIZES)} sizes)")

    if sys.platform != "darwin" or not shutil.which("iconutil"):
        print("  skipping .icns (needs macOS iconutil)")
        return

    # iconutil wants an .iconset directory with exact Apple filenames, including
    # the @2x retina variants.
    iconset = os.path.join(ASSETS, "icon.iconset")
    shutil.rmtree(iconset, ignore_errors=True)
    os.makedirs(iconset)
    for base in (16, 32, 128, 256, 512):
        renders[base].save(os.path.join(iconset, f"icon_{base}x{base}.png"))
        renders[base * 2].save(os.path.join(iconset, f"icon_{base}x{base}@2x.png"))

    subprocess.run(["iconutil", "-c", "icns", iconset,
                    "-o", os.path.join(ASSETS, "icon.icns")], check=True)
    shutil.rmtree(iconset)
    print("  wrote assets/icon.icns")


if __name__ == "__main__":
    main()
