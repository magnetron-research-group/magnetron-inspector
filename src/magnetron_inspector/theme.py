from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

BG = '#f6f6f3'
SURFACE = '#ffffff'
SURFACE_2 = '#fbfbf9'
SURFACE_3 = '#efeee9'
HEADER = '#f4f3ef'
BORDER = '#e3e2dc'
BORDER_STRONG = '#cbc9c0'
TEXT = '#1b1d22'
TEXT_DIM = '#5f6470'
TEXT_FAINT = '#9a9ea8'
INK = '#1b1d22'
ACCENT = '#2f5bd3'
ACCENT_HOVER = '#264bb3'
ACCENT_SOFT = '#e8eefc'
ACCENT_2 = '#0f8a7e'
SELECTION = '#dce5fb'
ALT_ROW = '#fafaf7'
GRIDLINE = '#eeede8'
WARN = '#a86a12'
ERROR = '#c53030'
OK = '#2b7a4b'

# Diverging RdBu ends for the tensor heatmap, NaN/Inf highlight
HEAT_NEG = '#2166ac'
HEAT_POS = '#b2182b'
NAN = '#d6336c'

# Muted categorical palette for dtype shares
CATEGORICAL = ('#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc948', '#b07aa1')

# JSON value type colors
JSON_COLORS: dict[str, str] = {
    'str': '#2f7d32',
    'int': '#b45309',
    'float': '#b45309',
    'bool': '#7c3aed',
    'null': '#8a8f99',
    'dict': '#1d4ed8',
    'list': '#0e7490',
    'json': '#be185d',
}


def _first_family(candidates: tuple[str, ...], fallback: str) -> str:
    families = set(QFontDatabase.families())
    return next((f for f in candidates if f in families), fallback)


def serif_family() -> str:
    return _first_family(('Source Serif 4', 'Source Serif Pro', 'IBM Plex Serif', 'Charter', 'Noto Serif', 'DejaVu Serif'), 'serif')


def sans_family() -> str:
    return _first_family(('Inter', 'IBM Plex Sans', 'Source Sans 3', 'Noto Sans', 'Cantarell', 'DejaVu Sans'), 'sans-serif')


def mono_font(size: int = 10) -> QFont:
    families = set(QFontDatabase.families())
    for fam in ('JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Ubuntu Mono', 'DejaVu Sans Mono'):
        if fam in families:
            f = QFont(fam, size)
            break
    else:
        f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        f.setPointSize(size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


_ASSETS = Path(__file__).parent.joinpath('assets').as_posix()

QSS_TEMPLATE = """
* {{ outline: none; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}
QSplitter::handle:hover {{ background: {SURFACE_3}; }}

QFrame#Panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#Card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel {{ background: transparent; }}
QLabel#Title {{ font-family: "{SERIF}"; font-size: 17pt; font-weight: 600; color: {INK}; }}
QLabel#Subtitle {{ color: {TEXT_DIM}; }}
QLabel#SectionTitle {{ color: {TEXT_DIM}; font-size: 8pt; font-weight: 700; letter-spacing: 1.2px; }}
QLabel#CardValue {{ font-family: "{SERIF}"; font-size: 15pt; font-weight: 600; color: {INK}; }}
QLabel#CardLabel {{ color: {TEXT_FAINT}; font-size: 8pt; font-weight: 600; letter-spacing: 0.8px; }}
QLabel#Badge {{ background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 4px; padding: 2px 8px; font-weight: 600; }}
QLabel#BadgeWarn {{ background: #fbf1df; color: {WARN}; border-radius: 4px; padding: 2px 8px; font-weight: 600; }}
QLabel#BadgeOk {{ background: #e5f3ea; color: {OK}; border-radius: 4px; padding: 2px 8px; font-weight: 600; }}
QLabel#Hint {{ color: {TEXT_FAINT}; }}

QToolBar {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; padding: 6px 10px; spacing: 6px; }}
QToolBar QLabel {{ color: {TEXT_DIM}; }}
QToolBar > QWidget {{ background: transparent; }}
QToolButton, QPushButton {{
    background: {SURFACE}; border: 1px solid {BORDER_STRONG}; border-radius: 6px; padding: 5px 12px; color: {TEXT};
}}
QToolButton:hover, QPushButton:hover {{ background: {SURFACE_3}; }}
QToolButton:pressed, QPushButton:pressed {{ background: {BORDER}; }}
QToolButton:disabled, QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; }}
QToolButton:checked, QPushButton:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#Primary {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: #ffffff; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; }}
QToolButton::menu-indicator {{ image: none; }}
QPushButton#Segment {{ padding: 4px 14px; border-radius: 6px; color: {TEXT_DIM}; font-weight: 600; }}
QPushButton#Segment:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT}; }}

QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{
    background: {SURFACE}; border: 1px solid {BORDER_STRONG}; border-radius: 6px; padding: 4px 8px;
    selection-background-color: {SELECTION}; selection-color: {TEXT};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {TEXT_FAINT}; background: {SURFACE_2}; }}
QPlainTextEdit {{ padding: 8px; background: {SURFACE_2}; border-color: {BORDER}; }}
QSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 18px; border: none; background: transparent; subcontrol-origin: border; }}
QSpinBox::up-button {{ subcontrol-position: top right; margin: 3px 3px 0 0; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 3px 3px 0; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {SURFACE_3}; border-radius: 4px; }}
QSpinBox::up-arrow {{ image: url({ASSETS}/up.svg); width: 9px; height: 6px; }}
QSpinBox::down-arrow {{ image: url({ASSETS}/down.svg); width: 9px; height: 6px; }}
QSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled {{ image: none; }}
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/down.svg); width: 9px; height: 6px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER}; selection-background-color: {SELECTION}; selection-color: {TEXT}; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid {BORDER_STRONG}; background: {SURFACE}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({ASSETS}/check.svg); }}
QCheckBox:disabled {{ color: {TEXT_FAINT}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; background: {BG}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM}; padding: 8px 16px; margin-right: 4px;
    border: none; border-bottom: 2px solid transparent; font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {ACCENT}; }}

QTreeView, QTableView, QListView {{
    background: {SURFACE}; alternate-background-color: {ALT_ROW}; border: none;
    selection-background-color: {SELECTION}; selection-color: {TEXT};
}}
QTreeView::item, QListView::item {{ padding: 3px 2px; }}
QTreeView::item:hover {{ background: {SURFACE_3}; }}
QTreeView::item:selected, QTableView::item:selected {{ background: {SELECTION}; color: {TEXT}; }}
QTableView {{ gridline-color: {GRIDLINE}; }}
QHeaderView {{ background: {HEADER}; border: none; }}
QHeaderView::section {{
    background: {HEADER}; color: {TEXT_DIM}; border: none; border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER}; padding: 4px 8px; font-weight: 600;
}}
QHeaderView::section:checked {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
QTableCornerButton::section {{ background: {HEADER}; border: none; border-bottom: 1px solid {BORDER}; border-right: 1px solid {BORDER}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle {{ background: #d3d1c8; border-radius: 3px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: #b9b7ad; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {SURFACE}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {TEXT_DIM}; padding: 0 6px; }}
QToolTip {{ background: {INK}; color: #ffffff; border: none; padding: 5px 7px; border-radius: 4px; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; padding: 4px; border-radius: 6px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {SELECTION}; color: {TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
"""


def stylesheet() -> str:
    tokens = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, str)}
    return QSS_TEMPLATE.format(**tokens, SERIF=serif_family(), ASSETS=_ASSETS)


def apply(app: QApplication) -> None:
    app.setStyle('Fusion')
    pal = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, BG),
        (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, SURFACE),
        (QPalette.ColorRole.AlternateBase, ALT_ROW),
        (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.Button, SURFACE_2),
        (QPalette.ColorRole.ButtonText, TEXT),
        (QPalette.ColorRole.Highlight, SELECTION),
        (QPalette.ColorRole.HighlightedText, TEXT),
        (QPalette.ColorRole.ToolTipBase, INK),
        (QPalette.ColorRole.ToolTipText, '#ffffff'),
        (QPalette.ColorRole.PlaceholderText, TEXT_FAINT),
    ):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setFont(QFont(sans_family(), 10))
    app.setStyleSheet(stylesheet())
