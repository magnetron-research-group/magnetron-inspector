from __future__ import annotations

from functools import cache
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QLinearGradient, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from . import theme

LOGO_PATH = Path(__file__).parent / 'assets' / 'logo.png'
ROUND_LOGO_PATH = Path(__file__).parent / 'assets' / 'logo-round.svg'
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
_SRC_STROKE = 13.0  # Stroke width of the source artwork in pixels
_MIN_STROKE = 1.5  # Thinnest stroke we allow after scaling, so small renders stay legible
_OVERSAMPLE = 8


def _ink_bounds(ink: QImage, threshold: int = 16) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of pixels darker than threshold in a Grayscale8 ink image."""
    hit = bytes(0 if i <= threshold else 1 for i in range(256))
    bits = ink.constBits()
    bits.setsize(ink.sizeInBytes())
    data, bpl, w = bytes(bits), ink.bytesPerLine(), ink.width()
    x0, x1, y0, y1 = w, -1, -1, -1
    for y in range(ink.height()):
        row = data[y * bpl : y * bpl + w].translate(hit)
        first = row.find(1)
        if first < 0:
            continue
        y0 = y if y0 < 0 else y0
        y1 = y
        x0, x1 = min(x0, first), max(x1, row.rfind(1))
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


@cache
def _mask() -> QImage:
    """The logo strokes as an alpha mask, cropped to their bounds. White fills become transparent."""
    src = QImage(str(LOGO_PATH))
    paper = QImage(src.size(), QImage.Format.Format_ARGB32)
    paper.fill(Qt.GlobalColor.white)
    p = QPainter(paper)
    p.drawImage(0, 0, src)
    p.end()
    ink = paper.convertToFormat(QImage.Format.Format_Grayscale8)
    ink.invertPixels()  # Darkness becomes coverage
    ink = ink.copy(*_ink_bounds(ink))
    mask = QImage(ink.size(), QImage.Format.Format_ARGB32)
    mask.fill(Qt.GlobalColor.white)
    mask.setAlphaChannel(ink)
    return mask


def _dilate(mask: QImage, radius: int) -> QImage:
    """Thicken strokes by stamping the mask at every offset inside a disc."""
    out = QImage(mask.width() + 2 * radius, mask.height() + 2 * radius, QImage.Format.Format_ARGB32_Premultiplied)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                p.drawImage(radius + dx, radius + dy, mask)
    p.end()
    return out


@cache
def _fitted_mask(px: int) -> QImage:
    """Mask scaled to fit px, with strokes thickened when plain downscaling would make them too thin."""
    src = _mask()
    longest = max(src.width(), src.height())
    mid = min(longest, px * _OVERSAMPLE)
    stroke_mid = _SRC_STROKE * mid / longest
    radius = max(0, round((_MIN_STROKE * mid / px - stroke_mid) / 2))
    fit = Qt.AspectRatioMode.KeepAspectRatio
    smooth = Qt.TransformationMode.SmoothTransformation
    mask = src.scaled(mid, mid, fit, smooth)
    if radius:
        mask = _dilate(mask, radius)
    return mask.scaled(px, px, fit, smooth)


def logo_pixmap(size: int, colors: tuple[str, str] = (theme.INK, theme.INK), dpr: float = 1.0) -> QPixmap:
    """Logo strokes filled with colors (a diagonal gradient when they differ), fitted into a size x size square."""
    px = max(1, round(size * dpr))
    mask = _fitted_mask(px)
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.drawImage((px - mask.width()) // 2, (px - mask.height()) // 2, mask)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    grad = QLinearGradient(0, 0, px, px)
    grad.setColorAt(0.0, QColor(colors[0]))
    grad.setColorAt(1.0, QColor(colors[1]))
    p.fillRect(img.rect(), grad)
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(dpr)
    return pm


def round_logo(size: int) -> QPixmap:
    """The round Magnetron logo from the magnetron repo (media/logo-border.svg), rendered as is."""
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    QSvgRenderer(str(ROUND_LOGO_PATH)).render(p, QRectF(0, 0, size, size))
    p.end()
    return QPixmap.fromImage(img)


@cache
def app_icon() -> QIcon:
    icon = QIcon()
    for s in ICON_SIZES:
        icon.addPixmap(round_logo(s))
    return icon
