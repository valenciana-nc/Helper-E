from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QFont, QFontDatabase, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import oauth_codex
import config
import theme
from conversation_store import StoredConversation, StoredMessage
from env_io import read_env, write_env

log = logging.getLogger("helper.dashboard")


SettingsSchema = list[tuple[str, list[tuple[str, str, str, Any, bool]]]]


def _default_schema() -> SettingsSchema:
    return [
        ("Providers & models", [
            ("HELPER_PROVIDER", "Provider", "provider", "codex", True),
            ("HELPER_AGENT_MODEL", "Computer-use model", "model", "gpt-5.5", True),
            ("HELPER_REASONING_MODEL", "Reasoning model", "model", "gpt-5.5", True),
            ("HELPER_API_BASE_URL", "OpenAI-compat base URL", "text", "", True),
            ("HELPER_API_KEY", "OpenAI-compat API key", "password", "", True),
            ("HELPER_ANTHROPIC_API_KEY", "Anthropic API key", "password", "", True),
            ("HELPER_GEMINI_API_KEY", "Gemini API key", "password", "", True),
            ("HELPER_STT_MODEL", "Speech-to-text (OpenAI)", "text", "whisper-1", True),
            ("HELPER_TTS_MODEL", "Text-to-speech (OpenAI)", "text", "gpt-4o-mini-tts", True),
            ("HELPER_TTS_VOICE", "TTS voice", "text", "alloy", True),
            ("HELPER_SPEAK_TYPED_CHAT", "Speak typed chat replies", "toggle", False, True),
        ]),
        ("Generation", [
            ("HELPER_TEMPERATURE", "Temperature", "slider_float", 0.7, False),
            ("HELPER_MAX_TOKENS", "Max output tokens", "int", 4096, False),
        ]),
        ("Agent", [
            ("HELPER_HOTKEY", "Hotkey", "text", "ctrl+shift+space", True),
            ("HELPER_DEFAULT_MODE", "Default mode", "choice", ["help", "active"], True),
            ("HELPER_ROUTE_CLASSIFIER", "Use LLM route classifier", "toggle", True, True),
            ("HELPER_MAX_AGENT_STEPS", "Max steps per task", "int", 25, True),
            ("HELPER_AGENT_TIMEOUT_SEC", "Task timeout (s)", "int", 180, True),
            ("HELPER_SCREENSHOT_MAX_EDGE", "Screenshot max edge", "int", 1280, True),
            ("HELPER_HISTORY_MAX_TURNS", "History max turns", "int", 12, True),
            ("HELPER_HISTORY_MAX_TOKENS", "History max tokens", "int", 12000, True),
        ]),
        ("Audio", [
            ("HELPER_AUDIO_SAMPLE_RATE", "Sample rate (Hz)", "int", 16000, True),
            ("HELPER_AUDIO_CHANNELS", "Channels", "int", 1, True),
            ("HELPER_AUDIO_BLOCKSIZE", "Block size", "int", 0, True),
            ("HELPER_AUDIO_SILENCE_THRESHOLD", "Silence threshold", "int", 700, True),
            ("HELPER_AUDIO_MIN_SECONDS", "Minimum seconds", "float", 0.25, True),
            ("HELPER_AUDIO_TRIM_PAD_MS", "Trim pad (ms)", "int", 150, True),
        ]),
        ("Appearance", [
            ("HELPER_THEME", "Theme", "theme", "light", True),
        ]),
    ]


# Section heading icons. First entry = Segoe Fluent Icons / MDL2 codepoint
# (used when the icon font is present), second = plain-Unicode fallback.
_SECTION_GLYPHS: dict[str, tuple[str, str]] = {
    "Providers & models": ("", "◆"),   # ServerNetwork → diamond
    "Generation":         ("", "✦"),   # Lightbulb → spark
    "Agent":              ("", "⚙"),   # Settings gear
    "Audio":              ("", "♪"),   # Volume → note
    "Appearance":         ("", "◐"),   # Color → half-circle
}


def _env_value(env_values: dict[str, str], key: str, default: str) -> str:
    # Model fields used to silently fall back when the value looked invalid
    # for the Codex OAuth path; that breaks the new multi-provider UI because
    # a user typing "claude-..." would see "gpt-5.5" reappear. We now return
    # whatever the user saved verbatim; validation happens at request time.
    if key in env_values and env_values[key] != "":
        return env_values[key]
    if key.startswith("HELPER_"):
        suffix = key.removeprefix("HELPER_")
        for legacy_key in (f"HELPLER_{suffix}", f"HARVIS_{suffix}"):
            if env_values.get(legacy_key, ""):
                return env_values[legacy_key]
    return default


# --- Palette + QSS sourced from theme.py --------------------------------------
# These names are kept for backwards-compat with the rest of this module. All
# values are derived from the active palette (config.THEME) at module load.

_PALETTE = theme.active_palette()


def _c(token: str) -> str:
    return _PALETTE[token]


_COLOR_BG_WINDOW = _c("bg_window")
_COLOR_BG_SIDEBAR = _c("bg_sidebar")
_COLOR_BG_TOPBAR = _c("bg_topbar")
_COLOR_BG_CARD = _c("bg_card")
_COLOR_BG_SUBTLE = _c("bg_subtle")
_COLOR_BG_HOVER = _c("bg_hover")
_COLOR_BG_NAV_SELECTED = _c("bg_nav_selected")

_COLOR_BORDER_LIGHT = _c("border_light")
_COLOR_BORDER_MID = _c("border_mid")

_COLOR_TEXT_PRIMARY = _c("text_primary")
_COLOR_TEXT_SECONDARY = _c("text_secondary")
_COLOR_TEXT_MUTED = _c("text_muted")
_COLOR_TEXT_PLACEHOLDER = _c("text_placeholder")

_COLOR_ACCENT = _c("accent")
_COLOR_ACCENT_HOVER = _c("accent_hover")
_COLOR_ACCENT_SUBTLE = _c("accent_subtle")


_FRAME_QSS = theme.qss(theme.FRAME_QSS, _PALETTE)
_SIDEBAR_QSS = theme.qss(theme.SIDEBAR_QSS, _PALETTE)
_NAV_BUTTON_QSS = theme.qss(theme.NAV_BUTTON_QSS, _PALETTE)
_BACK_BUTTON_QSS = theme.qss(theme.BACK_BUTTON_QSS, _PALETTE)
_INPUT_QSS = theme.qss(theme.INPUT_QSS, _PALETTE)
_BUTTON_QSS = theme.qss(theme.BUTTON_QSS, _PALETTE)
_GHOST_BUTTON_QSS = theme.qss(theme.GHOST_BUTTON_QSS, _PALETTE)
_SEGMENTED_BUTTON_QSS = theme.qss(theme.SEGMENTED_BUTTON_QSS, _PALETTE)
_SCROLL_QSS = theme.qss(theme.SCROLL_QSS, _PALETTE)
_LOG_QSS = theme.qss(theme.LOG_QSS, _PALETTE)
_CONVO_ROW_QSS = theme.qss(theme.CONVO_ROW_QSS, _PALETTE)
_STAT_CARD_QSS = theme.qss(theme.STAT_CARD_QSS, _PALETTE)
_CARD_QSS = theme.qss(theme.CARD_QSS, _PALETTE)
_STATUS_CARD_QSS = _CARD_QSS
_ACCOUNT_CARD_QSS = _CARD_QSS
_VOICE_CARD_QSS = _CARD_QSS
_FOOTER_BAR_QSS = theme.qss(theme.FOOTER_BAR_QSS, _PALETTE)
_PILL_QSS = theme.qss(theme.PILL_QSS, _PALETTE)


def _section_card_qss(section_name: str) -> str:
    del section_name  # all sections share the same card style now
    return _CARD_QSS


# Pick the best available Windows icon font once at module load. Segoe Fluent
# Icons ships with Windows 11; Segoe MDL2 Assets is on Windows 10. Both share
# the same Private Use Area codepoints we use below, so a single lookup table
# works for either. Fallback is plain Unicode glyphs in Segoe UI.
def _pick_icon_font() -> str | None:
    families = set(QFontDatabase.families())
    for candidate in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if candidate in families:
            return candidate
    return None


_ICON_FONT: str | None = None
_ICON_FONT_PROBED = False


def _icon_font() -> str | None:
    """Resolve the Windows icon font after QApplication exists."""
    global _ICON_FONT, _ICON_FONT_PROBED
    if not _ICON_FONT_PROBED:
        _ICON_FONT = _pick_icon_font()
        _ICON_FONT_PROBED = True
    return _ICON_FONT

# Codepoints from Segoe Fluent Icons / MDL2 Assets.
_NAV_ICON_GLYPHS: dict[str, str] = {
    "conversations": "",  # Comment / chat
    "settings": "",        # Settings gear
    "status": "",          # Health / status pulse
    "account": "",         # Contact / person
}

# Plain-Unicode fallback if the icon font isn't installed. Chosen so all four
# glyphs render at a comparable visual weight in Segoe UI.
_NAV_ICON_FALLBACK: dict[str, str] = {
    "conversations": "▢",
    "settings": "⚙",
    "status": "◉",
    "account": "◌",
}


def _render_icon(glyph: str, size: int, color_hex: str, *, use_icon_font: bool) -> QIcon:
    """Render a text glyph into a transparent QIcon of (size, size)."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    icon_font = _icon_font()
    if use_icon_font and icon_font:
        font = QFont(icon_font, max(8, size - 4))
    else:
        font = QFont("Segoe UI", max(8, size - 4))
    painter.setFont(font)

    # color_hex is an rgb(...) string; pull it apart for QColor.
    painter.setPen(QPen(_qcolor_from_token(color_hex)))
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return QIcon(pix)


def _qcolor_from_token(spec: str) -> QColor:
    """Parse the 'rgb(r, g, b)' / 'rgba(r, g, b, a)' strings used in the palette."""
    s = spec.strip()
    if s.startswith("rgb"):
        inner = s[s.find("(") + 1 : s.rfind(")")]
        parts = [p.strip() for p in inner.split(",")]
        try:
            r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
            a = int(parts[3]) if len(parts) > 3 else 255
            return QColor(r, g, b, a)
        except (ValueError, IndexError):
            return QColor(0, 0, 0)
    return QColor(s)


def _nav_icon(key: str, color_token: str = "text_secondary") -> QIcon:
    icon_font = _icon_font()
    glyph = _NAV_ICON_GLYPHS.get(key) if icon_font else None
    if not glyph:
        glyph = _NAV_ICON_FALLBACK.get(key, "•")
    return _render_icon(glyph, 18, _c(color_token), use_icon_font=bool(icon_font and _NAV_ICON_GLYPHS.get(key) == glyph))


# Curated provider menu used by the new provider/model dropdowns.
_PROVIDER_KEYS: tuple[str, ...] = ("codex", "openai_compat", "anthropic", "gemini")


def _provider_label(key: str) -> str:
    return config.PROVIDER_LABELS.get(key, key)


def _provider_key_from_label(label: str) -> str:
    for k in _PROVIDER_KEYS:
        if config.PROVIDER_LABELS.get(k, k) == label:
            return k
    return label  # treat as raw key


def _date_group_label(when: float) -> str:
    if when <= 0:
        return "EARLIER"
    today = date.today()
    day = datetime.fromtimestamp(when).date()
    if day == today:
        return "TODAY"
    if day == today - timedelta(days=1):
        return "YESTERDAY"
    if day.year == today.year:
        return day.strftime("%b %d").upper().replace(" 0", " ")
    return day.strftime("%b %d, %Y").upper().replace(" 0", " ")


def _format_time(when: float) -> str:
    if when <= 0:
        return "--:--"
    return datetime.fromtimestamp(when).strftime("%I:%M %p").lstrip("0")


def _day_streak(stored: list[StoredConversation]) -> int:
    if not stored:
        return 0
    days = {datetime.fromtimestamp(c.started_at).date() for c in stored if c.started_at > 0}
    if not days:
        return 0
    streak = 0
    cursor = date.today()
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


class TranscriptBubble(QFrame):
    def __init__(self, message: StoredMessage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        is_user = message.role == "user"
        label = QLabel(message.text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setFont(QFont("Segoe UI", 10))

        if is_user:
            label.setStyleSheet(
                f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
            )
            self.setStyleSheet(
                "TranscriptBubble {"
                f"  background: {_COLOR_BG_HOVER};"
                f"  border: 1px solid {_COLOR_BORDER_LIGHT};"
                "  border-radius: 14px;"
                "  border-bottom-right-radius: 4px;"
                "}"
            )
        else:
            label.setStyleSheet(
                f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
            )
            self.setStyleSheet(
                "TranscriptBubble {"
                f"  background: {_COLOR_BG_CARD};"
                f"  border: 1px solid {_COLOR_BORDER_LIGHT};"
                "  border-radius: 14px;"
                "  border-bottom-left-radius: 4px;"
                "}"
            )

        layout.addWidget(label)

        if message.timestamp > 0:
            ts = QLabel(_format_time(message.timestamp))
            ts.setFont(QFont("Segoe UI", 8))
            ts.setStyleSheet(
                f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
            )
            ts.setAlignment(
                Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft
            )
            layout.addWidget(ts)


class ConversationTranscriptDialog(QDialog):
    def __init__(self, conversation: StoredConversation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(conversation.title or "Conversation")
        self.setModal(True)
        self.resize(640, 580)
        self.setStyleSheet(f"background: {_COLOR_BG_WINDOW};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(68)
        header.setStyleSheet(
            f"background-color: {_COLOR_BG_TOPBAR}; "
            f"border-bottom: 1px solid {_COLOR_BORDER_LIGHT};"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 0, 14, 0)

        title_wrap = QVBoxLayout()
        title_wrap.setContentsMargins(0, 0, 0, 0)
        title_wrap.setSpacing(1)

        title = QLabel(conversation.title or "Conversation")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        title.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )

        when = datetime.fromtimestamp(conversation.started_at).strftime("%b %d, %Y · %I:%M %p")
        subtitle = QLabel(f"{when}  ·  {len(conversation.messages)} messages")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )

        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_GHOST_BUTTON_QSS)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)

        header_layout.addLayout(title_wrap)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(_SCROLL_QSS)
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(18, 18, 18, 18)
        body_layout.setSpacing(10)

        if not conversation.messages:
            empty = QLabel("This conversation has no messages.")
            empty.setStyleSheet(
                f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
            )
            body_layout.addWidget(empty)
        else:
            for message in conversation.messages:
                bubble = TranscriptBubble(message)
                row = QHBoxLayout()
                if message.role == "user":
                    row.addStretch()
                    row.addWidget(bubble, 5)
                else:
                    row.addWidget(bubble, 5)
                    row.addStretch()
                body_layout.addLayout(row)
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.accept()
            event.accept()
            return
        super().keyPressEvent(event)


class DashboardWindow(QWidget):
    closed = pyqtSignal()

    NAV_CONVERSATIONS = "conversations"
    NAV_SETTINGS = "settings"
    NAV_STATUS = "status"
    NAV_ACCOUNT = "account"
    NAV_ORDER = (NAV_CONVERSATIONS, NAV_SETTINGS, NAV_STATUS, NAV_ACCOUNT)

    def __init__(
        self,
        env_path: Path,
        status_provider: Callable[[], dict[str, Any]],
        restart_callback: Callable[[], None],
        auth_changed_callback: Callable[[bool], None] | None = None,
        conversations_provider: Callable[[], dict[str, Any]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._env_path = env_path
        self._status_provider = status_provider
        self._restart_callback = restart_callback
        self._auth_changed_callback = auth_changed_callback
        self._conversations_provider = conversations_provider
        self._schema = _default_schema()
        self._field_widgets: dict[str, QWidget] = {}
        self._field_restart: dict[str, bool] = {}
        self._original_values: dict[str, str] = {}
        self._field_label_widgets: dict[str, QLabel] = {}
        self._field_row_widgets: dict[str, QWidget] = {}
        self._field_password_toggles: dict[str, QPushButton] = {}
        self._section_pills: dict[str, QLabel] = {}
        self._provider_pill: QLabel | None = None
        self._nav_buttons: dict[str, QPushButton] = {}
        self._page_index: dict[str, int] = {}
        self._last_rendered_signature: tuple[Any, ...] | None = None

        self._build_window()
        self._build_ui()
        self._refresh_status()
        self._refresh_conversations(force=True)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.timeout.connect(self._refresh_conversations)

    def _build_window(self) -> None:
        self.setWindowTitle("Helper Dashboard")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet(
            f"background: {_COLOR_BG_WINDOW}; color: {_COLOR_TEXT_PRIMARY};"
        )
        self.setMinimumSize(760, 540)
        self.resize(960, 660)

    def _build_ui(self) -> None:
        self._frame = QFrame(self)
        self._frame.setObjectName("MainFrame")
        self._frame.setStyleSheet(_FRAME_QSS)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._frame)

        layout = QVBoxLayout(self._frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        body = QFrame()
        body.setStyleSheet("background: transparent;")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_sidebar())

        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {_COLOR_BG_WINDOW};")
        self._page_index[self.NAV_CONVERSATIONS] = self._stack.addWidget(self._build_conversations_page())
        self._page_index[self.NAV_SETTINGS] = self._stack.addWidget(self._build_settings_page())
        self._page_index[self.NAV_STATUS] = self._stack.addWidget(self._build_status_page())
        self._page_index[self.NAV_ACCOUNT] = self._stack.addWidget(self._build_account_page())
        body_layout.addWidget(self._stack, 1)

        layout.addWidget(body, 1)
        self._select_nav(self.NAV_CONVERSATIONS)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setStyleSheet(_SIDEBAR_QSS)
        sidebar.setFixedWidth(240)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(
            theme.SPACING.MD, theme.SPACING.LG, theme.SPACING.MD, theme.SPACING.MD
        )
        layout.setSpacing(2)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(theme.SPACING.SM, 0, theme.SPACING.SM, 0)
        brand_row.setSpacing(theme.SPACING.MD)

        logo_path = Path(__file__).parent / "assets" / "helper_logo.png"
        logo_label = QLabel()
        logo_label.setFixedSize(28, 28)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("background: transparent; border: none;")
        loaded_pixmap = False
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                pix = pix.scaled(
                    QSize(28, 28),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                logo_label.setPixmap(pix)
                loaded_pixmap = True
        if not loaded_pixmap:
            # Render a clean accent-colored mark when there's no logo file.
            icon_font = _icon_font()
            mark_glyph = "" if icon_font else "◆"
            logo_label.setText(mark_glyph)
            font = QFont(icon_font, 16) if icon_font else QFont("Segoe UI", 16, QFont.Weight.DemiBold)
            logo_label.setFont(font)
            logo_label.setStyleSheet(
                f"color: {_COLOR_ACCENT}; background: transparent; border: none;"
            )
        brand_row.addWidget(logo_label)

        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(-2)

        brand = QLabel("Helper")
        brand.setObjectName("SidebarBrand")
        brand.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        brand_text.addWidget(brand)

        brand_sub = QLabel("Desktop assistant")
        brand_sub.setObjectName("SidebarSubtitle")
        brand_sub.setFont(QFont("Segoe UI", 8))
        brand_text.addWidget(brand_sub)

        brand_row.addLayout(brand_text)
        brand_row.addStretch()
        layout.addLayout(brand_row)

        layout.addSpacing(theme.SPACING.LG)

        nav_group = QButtonGroup(sidebar)
        nav_group.setExclusive(True)

        for key, label, icon_key in (
            (self.NAV_CONVERSATIONS, "Conversations", "conversations"),
            (self.NAV_SETTINGS, "Settings", "settings"),
            (self.NAV_STATUS, "Status", "status"),
            (self.NAV_ACCOUNT, "Account", "account"),
        ):
            btn = QPushButton(label)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setStyleSheet(_NAV_BUTTON_QSS)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setIcon(_nav_icon(icon_key, "text_secondary"))
            btn.setIconSize(QSize(18, 18))
            btn.setMinimumHeight(34)
            btn.setProperty("nav_icon_key", icon_key)
            btn.toggled.connect(
                lambda checked, b=btn, k=icon_key: b.setIcon(
                    _nav_icon(k, "text_primary" if checked else "text_secondary")
                )
            )
            btn.clicked.connect(lambda _checked=False, target=key: self._select_nav(target))
            nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

        self._sidebar_foot_pill = self._make_status_pill("Not signed in", "warning")
        self._sidebar_foot_pill.setWordWrap(True)
        self._sidebar_foot_pill.setMaximumHeight(80)
        self._sidebar_foot_pill.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self._sidebar_foot_pill)

        # Keep the legacy attribute name available so existing _refresh_status
        # code paths that reference it don't NameError.
        self._sidebar_foot_chip = self._sidebar_foot_pill

        version_label = QLabel("Helper · local build")
        version_label.setFont(QFont("Segoe UI", 8))
        version_label.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none; padding-left: 4px;"
        )
        layout.addWidget(version_label)

        return sidebar

    def _select_nav(self, key: str) -> None:
        if key not in self._page_index:
            return
        for nav_key, btn in self._nav_buttons.items():
            btn.setChecked(nav_key == key)
        self._stack.setCurrentIndex(self._page_index[key])
        if key == self.NAV_CONVERSATIONS:
            self._refresh_conversations(force=True)
        elif key == self.NAV_STATUS:
            self._refresh_status()
        elif key == self.NAV_ACCOUNT:
            self._refresh_account_card()

    # --- Conversations page -------------------------------------------------

    def _build_conversations_page(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        outer = QHBoxLayout(container)
        outer.setContentsMargins(28, 28, 28, 20)
        outer.setSpacing(18)

        feed_wrap = QVBoxLayout()
        feed_wrap.setContentsMargins(0, 0, 0, 0)
        feed_wrap.setSpacing(10)

        heading_row = QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Conversations")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.DemiBold))
        title.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )

        self._convo_count_label = QLabel("0")
        self._convo_count_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self._convo_count_label.setStyleSheet(
            f"color: {_COLOR_TEXT_SECONDARY}; background: {_COLOR_BG_HOVER}; "
            f"border: 1px solid {_COLOR_BORDER_LIGHT}; border-radius: 10px; padding: 2px 10px;"
        )

        heading_row.addWidget(title)
        heading_row.addSpacing(8)
        heading_row.addWidget(self._convo_count_label)
        heading_row.addStretch()

        feed_wrap.addLayout(heading_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(_SCROLL_QSS)

        self._convo_body = QWidget()
        self._convo_body.setStyleSheet("background: transparent;")
        self._convo_body_layout = QVBoxLayout(self._convo_body)
        self._convo_body_layout.setContentsMargins(0, 4, 4, 4)
        self._convo_body_layout.setSpacing(6)
        self._convo_body_layout.addStretch()
        scroll.setWidget(self._convo_body)
        feed_wrap.addWidget(scroll, 1)

        outer.addLayout(feed_wrap, 7)
        outer.addWidget(self._build_stats_panel(), 3)
        return container

    def _build_stats_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)

        heading = QLabel("Stats")
        heading.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        heading.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        layout.addWidget(heading)

        self._stat_total = self._make_stat_card("Total conversations", "0")
        self._stat_messages = self._make_stat_card("Total messages", "0")
        self._stat_streak = self._make_stat_card("Day streak", "0")
        layout.addWidget(self._stat_total)
        layout.addWidget(self._stat_messages)
        layout.addWidget(self._stat_streak)
        layout.addStretch()
        return panel

    def _make_stat_card(self, label_text: str, initial_value: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_STAT_CARD_QSS)
        inner = QVBoxLayout(card)
        inner.setContentsMargins(16, 14, 16, 14)
        inner.setSpacing(2)

        value_label = QLabel(initial_value)
        value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.DemiBold))
        value_label.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        inner.addWidget(value_label)

        caption = QLabel(label_text)
        caption.setFont(QFont("Segoe UI", 9))
        caption.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        inner.addWidget(caption)

        card.setProperty("value_label", value_label)
        return card

    def _set_stat(self, card: QFrame, value: str) -> None:
        label = card.property("value_label")
        if isinstance(label, QLabel):
            label.setText(value)

    def _refresh_conversations(self, force: bool = False) -> None:
        if not hasattr(self, "_convo_body_layout"):
            return
        snapshot = self._conversations_snapshot()
        signature = self._snapshot_signature(snapshot)
        if not force and signature == self._last_rendered_signature:
            return
        self._last_rendered_signature = signature
        self._populate_conversations(snapshot)

    def _conversations_snapshot(self) -> dict[str, Any]:
        if self._conversations_provider is None:
            return {"stored": [], "live": None}
        try:
            payload = self._conversations_provider() or {}
        except Exception:
            log.exception("Conversation provider failed")
            return {"stored": [], "live": None}
        stored = payload.get("stored") or []
        if not isinstance(stored, list):
            stored = []
        live = payload.get("live")
        return {"stored": stored, "live": live if isinstance(live, StoredConversation) else None}

    def _snapshot_signature(self, snapshot: dict[str, Any]) -> tuple[Any, ...]:
        stored: list[StoredConversation] = snapshot.get("stored") or []
        live: StoredConversation | None = snapshot.get("live")
        stored_sig = tuple((c.id, c.started_at, c.ended_at, len(c.messages)) for c in stored)
        live_sig = (
            (live.id, live.started_at, live.ended_at, len(live.messages))
            if live is not None
            else None
        )
        return (stored_sig, live_sig)

    def _populate_conversations(self, snapshot: dict[str, Any]) -> None:
        layout = self._convo_body_layout
        self._clear_layout(layout)

        stored: list[StoredConversation] = snapshot.get("stored") or []
        live: StoredConversation | None = snapshot.get("live")

        total_messages = sum(len(c.messages) for c in stored) + (len(live.messages) if live else 0)
        total_count = len(stored) + (1 if live else 0)
        self._convo_count_label.setText(str(total_count))
        self._set_stat(self._stat_total, str(total_count))
        self._set_stat(self._stat_messages, str(total_messages))
        streak_input = list(stored)
        if live is not None:
            streak_input.append(live)
        self._set_stat(self._stat_streak, str(_day_streak(streak_input)))

        if total_count == 0:
            empty = QLabel(
                "No conversations yet.\n\n"
                "Open the chat bar on the orb (or tap the hotkey) and ask Helper anything. "
                "Each app session is saved here when you quit Helper."
            )
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                f"color: {_COLOR_TEXT_MUTED}; background: {_COLOR_BG_CARD}; "
                f"border: 1px dashed {_COLOR_BORDER_LIGHT}; border-radius: 12px; padding: 40px;"
            )
            layout.addWidget(empty)
            layout.addStretch()
            return

        rendered_groups: set[str] = set()
        if live is not None:
            self._add_group_header(layout, "TODAY", rendered_groups)
            layout.addWidget(self._make_conversation_row(live, is_live=True))

        for conversation in stored:
            label = _date_group_label(conversation.started_at)
            self._add_group_header(layout, label, rendered_groups)
            layout.addWidget(self._make_conversation_row(conversation, is_live=False))

        layout.addStretch()

    def _add_group_header(self, layout: QVBoxLayout, label: str, rendered: set[str]) -> None:
        if label in rendered:
            return
        rendered.add(label)
        header = QLabel(label)
        header.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        header.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none; "
            "letter-spacing: 1.2px; padding-top: 10px; padding-bottom: 4px;"
        )
        layout.addWidget(header)

    def _make_conversation_row(self, conversation: StoredConversation, is_live: bool) -> QPushButton:
        time_str = _format_time(conversation.started_at)
        title = conversation.title or conversation.derive_title() or "Conversation"
        live_tag = "  ·  LIVE" if is_live else ""
        text = f"{time_str:>8}    {title}{live_tag}"
        button = QPushButton(text)
        button.setObjectName("ConvoRow")
        button.setStyleSheet(_CONVO_ROW_QSS)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(46)
        captured = conversation
        button.clicked.connect(lambda _checked=False, c=captured: self._open_transcript(c))
        return button

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                continue
            sub = item.layout()
            if sub is not None:
                self._clear_layout(sub)

    def _open_transcript(self, conversation: StoredConversation) -> None:
        dialog = ConversationTranscriptDialog(conversation, parent=self)
        dialog.exec()

    # --- Settings page ------------------------------------------------------

    def _build_settings_page(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(_SCROLL_QSS)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(
            theme.SPACING.XL + 4, theme.SPACING.XL + 4, theme.SPACING.XL + 4, theme.SPACING.LG
        )
        body_layout.setSpacing(theme.SPACING.LG)

        body_layout.addWidget(
            self._build_page_heading("Settings", "Providers, generation, agent behavior, audio, appearance")
        )

        env_values = read_env(self._env_path)

        for section_name, fields in self._schema:
            body_layout.addWidget(self._build_section(section_name, fields, env_values))

        body_layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        outer.addWidget(self._build_footer())

        # Hook provider → model + API-key visibility wiring once all fields
        # are in the registry.
        self._wire_provider_dependencies()
        return container

    # ------------------------------------------------------------------
    # Provider <-> model + API key visibility wiring.

    def _wire_provider_dependencies(self) -> None:
        provider_widget = self._field_widgets.get("HELPER_PROVIDER")
        if not isinstance(provider_widget, QComboBox):
            return

        # Cache the section pill for the provider section so we can flip it on/off.
        self._provider_pill = self._section_pills.get("Providers & models")

        def _on_provider_changed(_idx: int = 0) -> None:
            self._refresh_provider_dependents()

        provider_widget.currentIndexChanged.connect(_on_provider_changed)

        # Also refresh whenever an API-key field changes, since validation
        # depends on whether the right one is populated.
        for key in (
            "HELPER_API_BASE_URL",
            "HELPER_API_KEY",
            "HELPER_ANTHROPIC_API_KEY",
            "HELPER_GEMINI_API_KEY",
        ):
            widget = self._field_widgets.get(key)
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(lambda _t: self._refresh_provider_dependents())

        self._refresh_provider_dependents()

    def _current_provider_key(self) -> str:
        widget = self._field_widgets.get("HELPER_PROVIDER")
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            if isinstance(data, str) and data:
                return data
        return "codex"

    def _refresh_provider_dependents(self) -> None:
        provider_key = self._current_provider_key()
        models = config.PROVIDER_MODELS.get(provider_key, [])

        # Repopulate model combos for both agent + reasoning fields.
        for model_key in ("HELPER_AGENT_MODEL", "HELPER_REASONING_MODEL"):
            combo = self._field_widgets.get(model_key)
            if not isinstance(combo, QComboBox):
                continue
            current_text = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(models)
            if current_text:
                combo.setEditText(current_text)
            elif models:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        # API-key visibility per provider.
        visibility = {
            "HELPER_API_BASE_URL": provider_key == "openai_compat",
            "HELPER_API_KEY": provider_key == "openai_compat",
            "HELPER_ANTHROPIC_API_KEY": provider_key == "anthropic",
            "HELPER_GEMINI_API_KEY": provider_key == "gemini",
        }
        for env_key, visible in visibility.items():
            self._set_field_visible(env_key, visible)

        # Pill: report sign-in / API-key readiness for the chosen provider.
        if self._provider_pill is None:
            return

        if provider_key == "codex":
            if oauth_codex.is_signed_in():
                self._set_pill(self._provider_pill, "Signed in", "success")
            else:
                self._set_pill(self._provider_pill, "Sign-in required", "warning")
        elif provider_key == "openai_compat":
            base = self._field_value("HELPER_API_BASE_URL")
            key = self._field_value("HELPER_API_KEY")
            if base and key:
                self._set_pill(self._provider_pill, "Ready", "success")
            else:
                self._set_pill(self._provider_pill, "Base URL + key required", "warning")
        elif provider_key == "anthropic":
            if self._field_value("HELPER_ANTHROPIC_API_KEY"):
                self._set_pill(self._provider_pill, "Ready", "success")
            else:
                self._set_pill(self._provider_pill, "API key required", "warning")
        elif provider_key == "gemini":
            if self._field_value("HELPER_GEMINI_API_KEY"):
                self._set_pill(self._provider_pill, "Ready", "success")
            else:
                self._set_pill(self._provider_pill, "API key required", "warning")
        else:
            self._set_pill(self._provider_pill, "", "neutral")

    def _set_field_visible(self, env_key: str, visible: bool) -> None:
        label = self._field_label_widgets.get(env_key)
        row_widget = self._field_row_widgets.get(env_key)
        if label is None or row_widget is None:
            return
        label.setVisible(visible)
        row_widget.setVisible(visible)

    def _field_value(self, env_key: str) -> str:
        widget = self._field_widgets.get(env_key)
        if widget is None:
            return ""
        return self._widget_value(widget)

    def _build_page_heading(self, title: str, subtitle: str = "") -> QWidget:
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.DemiBold))
        title_label.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        layout.addWidget(title_label)

        if subtitle:
            sub_label = QLabel(subtitle)
            sub_label.setFont(QFont("Segoe UI", 9))
            sub_label.setStyleSheet(
                f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
            )
            layout.addWidget(sub_label)
        return wrap

    def _build_section(
        self,
        section_name: str,
        fields: list[tuple[str, str, str, Any, bool]],
        env_values: dict[str, str],
    ) -> QFrame:
        section = QFrame()
        section.setStyleSheet(_section_card_qss(section_name))
        outer = QVBoxLayout(section)
        outer.setContentsMargins(
            theme.SPACING.LG, theme.SPACING.MD, theme.SPACING.LG, theme.SPACING.LG
        )
        outer.setSpacing(theme.SPACING.MD)

        heading_row = QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)
        heading_row.setSpacing(8)

        glyph_pair = _SECTION_GLYPHS.get(section_name)
        if glyph_pair:
            primary, fallback = glyph_pair
            icon_font = _icon_font()
            if icon_font and primary:
                glyph_label = QLabel(primary)
                glyph_label.setFont(QFont(icon_font, 14))
            else:
                glyph_label = QLabel(fallback)
                glyph_label.setFont(QFont("Segoe UI", 12))
            glyph_label.setFixedWidth(20)
            glyph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            glyph_label.setStyleSheet(
                f"color: {_COLOR_ACCENT}; background: transparent; border: none;"
            )
            heading_row.addWidget(glyph_label)

        heading = QLabel(section_name)
        heading.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        heading.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; border: none; background: transparent;"
        )
        heading_row.addWidget(heading)
        heading_row.addStretch(1)

        # Section-level status pill (e.g. provider readiness). The actual text
        # is set later by the dependency wiring; default to hidden.
        section_pill = self._make_status_pill("", "neutral")
        section_pill.setVisible(False)
        heading_row.addWidget(section_pill)
        self._section_pills[section_name] = section_pill

        outer.addLayout(heading_row)

        # Subtle divider under the heading for visual rhythm.
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            f"background: {_COLOR_BORDER_LIGHT}; border: none;"
        )
        outer.addWidget(divider)

        form = QFormLayout()
        form.setContentsMargins(0, theme.SPACING.SM, 0, 0)
        form.setHorizontalSpacing(theme.SPACING.LG)
        form.setVerticalSpacing(theme.SPACING.MD)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        for env_key, label, kind, default, requires_restart in fields:
            fallback = str(default) if not isinstance(default, list) else default[0]
            if isinstance(default, bool):
                fallback = "true" if default else "false"
            current = _env_value(env_values, env_key, fallback)
            widget = self._make_widget(kind, default, current)
            self._field_widgets[env_key] = widget
            self._field_restart[env_key] = requires_restart
            self._original_values[env_key] = self._widget_value(widget)

            label_widget = QLabel(label + (" *" if requires_restart else ""))
            label_widget.setStyleSheet(
                f"color: {_COLOR_TEXT_SECONDARY}; background: transparent; border: none;"
            )
            label_widget.setFont(QFont("Segoe UI", 9))

            # Password fields get a small "Show" toggle next to them.
            if widget.property("kind") == "password":
                row_widget = self._wrap_password_row(env_key, widget)
            else:
                row_widget = widget
            form.addRow(label_widget, row_widget)
            self._field_label_widgets[env_key] = label_widget
            self._field_row_widgets[env_key] = row_widget

        outer.addLayout(form)
        return section

    def _wrap_password_row(self, env_key: str, line: QLineEdit) -> QWidget:
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING.SM)
        row.addWidget(line, 1)
        show_btn = QPushButton("Show")
        show_btn.setCheckable(True)
        show_btn.setStyleSheet(_GHOST_BUTTON_QSS)
        show_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        def _toggle(checked: bool, w: QLineEdit = line, b: QPushButton = show_btn) -> None:
            w.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
            b.setText("Hide" if checked else "Show")

        show_btn.toggled.connect(_toggle)
        row.addWidget(show_btn)
        self._field_password_toggles[env_key] = show_btn
        return wrap

    def _make_status_pill(self, text: str, kind: str) -> QLabel:
        pill = QLabel(text)
        pill.setObjectName("StatusPill")
        pill.setProperty("kind", kind)
        pill.setStyleSheet(_PILL_QSS)
        pill.setFont(QFont("Segoe UI", 9))
        pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pill.setMaximumHeight(22)
        return pill

    def _set_pill(self, pill: QLabel, text: str, kind: str) -> None:
        pill.setText(text)
        pill.setProperty("kind", kind)
        # Re-apply the stylesheet so the new "kind" property selector takes effect.
        pill.setStyleSheet(_PILL_QSS)
        pill.setVisible(bool(text))

    def _make_widget(self, kind: str, default: Any, current: str) -> QWidget:
        if kind == "text":
            w = QLineEdit()
            w.setStyleSheet(_INPUT_QSS)
            w.setText(current)
            return w
        if kind == "password":
            w = QLineEdit()
            w.setStyleSheet(_INPUT_QSS)
            w.setEchoMode(QLineEdit.EchoMode.Password)
            w.setText(current)
            # Stash echo-mode behavior on the widget so the section can wire
            # up a "Show" button next to it without keeping its own state.
            w.setProperty("kind", "password")
            return w
        if kind == "int":
            w = QSpinBox()
            w.setStyleSheet(_INPUT_QSS)
            w.setRange(0, 1_000_000)
            try:
                w.setValue(int(current))
            except ValueError:
                w.setValue(int(default))
            return w
        if kind == "float":
            w = QDoubleSpinBox()
            w.setStyleSheet(_INPUT_QSS)
            w.setRange(0.0, 10_000.0)
            w.setDecimals(3)
            w.setSingleStep(0.05)
            try:
                w.setValue(float(current))
            except ValueError:
                w.setValue(float(default))
            return w
        if kind == "slider_float":
            # Compound widget: slider + spinbox, kept in sync. The outer
            # QWidget exposes a `spin` property so _widget_value can read it.
            wrap = QWidget()
            wrap.setStyleSheet("background: transparent;")
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(theme.SPACING.SM)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 200)
            slider.setSingleStep(5)
            slider.setPageStep(10)
            slider.setMinimumWidth(140)

            spin = QDoubleSpinBox()
            spin.setStyleSheet(_INPUT_QSS)
            spin.setRange(0.0, 2.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)

            try:
                value = float(current)
            except (TypeError, ValueError):
                value = float(default)
            value = max(0.0, min(2.0, value))
            spin.setValue(value)
            slider.setValue(int(round(value * 100)))

            slider.valueChanged.connect(lambda v, s=spin: s.setValue(v / 100.0))
            spin.valueChanged.connect(lambda v, s=slider: s.setValue(int(round(v * 100))))

            row.addWidget(slider, 1)
            row.addWidget(spin, 0)
            wrap.setProperty("spin", spin)
            wrap.setProperty("kind", "slider_float")
            return wrap
        if kind == "toggle":
            w = QCheckBox()
            w.setStyleSheet(
                "QCheckBox { color: " + _COLOR_TEXT_SECONDARY + "; spacing: 8px; }"
                "QCheckBox::indicator { width: 16px; height: 16px;"
                " border: 1px solid " + _COLOR_BORDER_MID + ";"
                " border-radius: 4px; background: " + _COLOR_BG_CARD + "; }"
                "QCheckBox::indicator:checked { background: " + _COLOR_ACCENT + ";"
                " border: 1px solid " + _COLOR_ACCENT + "; }"
            )
            truthy = str(current).strip().lower() in {"1", "true", "yes", "on"}
            if not current:
                truthy = bool(default) if isinstance(default, bool) else False
            w.setChecked(truthy)
            w.setProperty("kind", "toggle")
            return w
        if kind == "choice":
            w = QComboBox()
            w.setStyleSheet(_INPUT_QSS)
            choices = list(default)
            w.addItems(choices)
            if current in choices:
                w.setCurrentIndex(choices.index(current))
            return w
        if kind == "provider":
            w = QComboBox()
            w.setStyleSheet(_INPUT_QSS)
            for key in _PROVIDER_KEYS:
                w.addItem(_provider_label(key), userData=key)
            target = (current or default or "codex").strip().lower() or "codex"
            idx = max(0, _PROVIDER_KEYS.index(target) if target in _PROVIDER_KEYS else 0)
            w.setCurrentIndex(idx)
            w.setProperty("kind", "provider")
            return w
        if kind == "model":
            # Editable combo populated dynamically by the provider's selection.
            w = QComboBox()
            w.setStyleSheet(_INPUT_QSS)
            w.setEditable(True)
            w.lineEdit().setPlaceholderText("Select or type a model name…")
            if current:
                w.setEditText(current)
            elif default:
                w.setEditText(str(default))
            w.setProperty("kind", "model")
            return w
        if kind == "theme":
            wrap = QWidget()
            wrap.setStyleSheet("background: transparent;")
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            group = QButtonGroup(wrap)
            group.setExclusive(True)

            chosen = (current or default or "light").strip().lower()
            if chosen not in ("light", "dark"):
                chosen = "light"

            for option in ("light", "dark"):
                btn = QPushButton(option.capitalize())
                btn.setCheckable(True)
                btn.setStyleSheet(_SEGMENTED_BUTTON_QSS)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setProperty("themeKey", option)
                if option == chosen:
                    btn.setChecked(True)
                group.addButton(btn)
                row.addWidget(btn)
            row.addStretch(1)
            wrap.setProperty("kind", "theme")
            wrap.setProperty("group", group)
            return wrap
        raise ValueError(f"Unknown field kind: {kind}")

    @staticmethod
    def _widget_value(widget: QWidget) -> str:
        kind = widget.property("kind") if widget is not None else None
        if kind == "slider_float":
            spin = widget.property("spin")
            if isinstance(spin, QDoubleSpinBox):
                return f"{spin.value():g}"
            return ""
        if kind == "toggle":
            assert isinstance(widget, QCheckBox)
            return "true" if widget.isChecked() else "false"
        if kind == "provider":
            assert isinstance(widget, QComboBox)
            data = widget.currentData()
            if isinstance(data, str) and data:
                return data
            return _provider_key_from_label(widget.currentText())
        if kind == "model":
            assert isinstance(widget, QComboBox)
            return widget.currentText().strip()
        if kind == "theme":
            group = widget.property("group")
            if group is not None:
                checked = group.checkedButton()
                if checked is not None:
                    key = checked.property("themeKey")
                    if isinstance(key, str):
                        return key
            return "light"
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        if isinstance(widget, QSpinBox):
            return str(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return f"{widget.value():g}"
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return ""

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setStyleSheet(_FOOTER_BAR_QSS)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(
            theme.SPACING.XL, theme.SPACING.MD, theme.SPACING.XL, theme.SPACING.MD
        )
        layout.setSpacing(theme.SPACING.MD)

        hint = QLabel("Fields marked * require a restart to take effect.")
        hint.setFont(QFont("Segoe UI", 8))
        hint.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        layout.addWidget(hint)

        layout.addStretch(1)

        self._save_status = self._make_status_pill("", "neutral")
        self._save_status.setVisible(False)
        layout.addWidget(self._save_status)

        save_button = QPushButton("Save")
        save_button.setStyleSheet(_GHOST_BUTTON_QSS)
        save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_button.clicked.connect(self._save_settings)
        layout.addWidget(save_button)

        restart_button = QPushButton("Save & restart")
        restart_button.setStyleSheet(_BUTTON_QSS)
        restart_button.setCursor(Qt.CursorShape.PointingHandCursor)
        restart_button.clicked.connect(self._save_and_restart)
        layout.addWidget(restart_button)

        return footer

    def _save_settings(self) -> tuple[dict[str, str], bool]:
        updates: dict[str, str] = {}
        needs_restart = False
        for env_key, widget in self._field_widgets.items():
            value = self._widget_value(widget)
            if value != self._original_values.get(env_key, ""):
                updates[env_key] = value
                if self._field_restart.get(env_key, False):
                    needs_restart = True

        # Preflight validation — block obvious provider misconfigurations.
        validation_error = self._validate_settings(updates)
        if validation_error is not None:
            self._set_pill(self._save_status, validation_error, "danger")
            return {}, False

        if not updates:
            self._set_pill(self._save_status, "No changes", "neutral")
            return updates, False

        try:
            write_env(self._env_path, updates)
        except Exception as exc:
            log.exception("Saving .env failed")
            self._set_pill(self._save_status, f"Save failed: {exc}", "danger")
            return updates, False

        for env_key in updates:
            self._original_values[env_key] = updates[env_key]

        message = f"Saved {len(updates)} change{'s' if len(updates) != 1 else ''}"
        kind = "success"
        if needs_restart:
            message += " · restart to apply"
            kind = "info"
        self._set_pill(self._save_status, message, kind)
        return updates, needs_restart

    def _validate_settings(self, _updates: dict[str, str]) -> str | None:
        """Return an error message if the *current* form state is invalid."""
        provider = self._current_provider_key()

        if provider == "openai_compat":
            base = self._field_value("HELPER_API_BASE_URL")
            key = self._field_value("HELPER_API_KEY")
            if not base or not key:
                return "OpenAI-compatible needs base URL + API key"
            base_error = config.validate_api_base_url(base)
            if base_error:
                return base_error
        elif provider == "anthropic":
            if not self._field_value("HELPER_ANTHROPIC_API_KEY"):
                return "Anthropic provider needs an API key"
        elif provider == "gemini":
            if not self._field_value("HELPER_GEMINI_API_KEY"):
                return "Gemini provider needs an API key"

        # Generation bounds (clamped on read in config.py, but flag here too).
        temp_raw = self._field_value("HELPER_TEMPERATURE")
        if temp_raw:
            try:
                temp = float(temp_raw)
                if temp < 0.0 or temp > 2.0:
                    return "Temperature must be between 0.0 and 2.0"
            except ValueError:
                return "Temperature must be a number"

        max_tokens_raw = self._field_value("HELPER_MAX_TOKENS")
        if max_tokens_raw:
            try:
                mt = int(float(max_tokens_raw))
                if mt < 1 or mt > 200_000:
                    return "Max tokens must be between 1 and 200000"
            except ValueError:
                return "Max tokens must be a number"

        return None

    def _save_and_restart(self) -> None:
        _, _ = self._save_settings()
        self._restart_callback()

    # --- Status page --------------------------------------------------------

    def _build_status_page(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 28, 28, 20)
        layout.setSpacing(14)

        layout.addWidget(self._build_page_heading("Status", "Backend health and logs"))

        info_card = QFrame()
        info_card.setStyleSheet(_STATUS_CARD_QSS)
        info_layout = QFormLayout(info_card)
        info_layout.setContentsMargins(14, 12, 14, 14)
        info_layout.setHorizontalSpacing(14)
        info_layout.setVerticalSpacing(8)

        self._status_labels: dict[str, QLabel] = {}
        for key, label in [
            ("backend", "Backend"),
            ("account", "Account"),
            ("agent", "Agent"),
            ("mode", "Current mode"),
            ("hotkey", "Hotkey"),
            ("error", "Last error"),
            ("log_path", "Log file"),
        ]:
            name = QLabel(label)
            name.setFont(QFont("Segoe UI", 9))
            name.setStyleSheet(
                f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
            )
            value = QLabel("—")
            value.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            value.setStyleSheet(
                f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
            )
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info_layout.addRow(name, value)
            self._status_labels[key] = value
        layout.addWidget(info_card)

        log_heading = QLabel("Recent log")
        log_heading.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        log_heading.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        layout.addWidget(log_heading)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet(_LOG_QSS)
        layout.addWidget(self._log_view, 1)
        return container

    def _refresh_status(self) -> None:
        if not hasattr(self, "_status_labels"):
            return
        try:
            snapshot = self._status_provider() or {}
        except Exception:
            log.exception("Status provider raised")
            return

        if snapshot.get("backend_ready"):
            backend_text = "Ready"
        elif snapshot.get("backend_initializing"):
            backend_text = "Starting..."
        elif snapshot.get("backend_error"):
            backend_text = f"Failed: {snapshot['backend_error']}"
        else:
            backend_text = "Not started"

        account_text = "Signed in" if snapshot.get("signed_in") else "Signed out"
        if snapshot.get("agent_ready"):
            agent_text = "Ready"
        elif snapshot.get("agent_error"):
            agent_text = f"Disabled: {snapshot['agent_error']}"
        elif snapshot.get("signed_in"):
            agent_text = "Not started"
        else:
            agent_text = "Disabled until sign-in"

        self._status_labels["backend"].setText(backend_text)
        self._status_labels["account"].setText(account_text)
        self._status_labels["agent"].setText(agent_text)
        self._status_labels["mode"].setText(str(snapshot.get("mode", "-")).upper())
        self._status_labels["hotkey"].setText(str(snapshot.get("hotkey", "-")))
        self._status_labels["error"].setText(
            str(snapshot.get("backend_error") or snapshot.get("agent_error") or "None")
        )

        log_path = snapshot.get("log_path")
        if isinstance(log_path, Path):
            self._status_labels["log_path"].setText(str(log_path))
            self._update_log_view(log_path)
        else:
            self._status_labels["log_path"].setText("—")

        if hasattr(self, "_sidebar_foot_pill"):
            if snapshot.get("signed_in"):
                email = oauth_codex.account_email() or "ChatGPT"
                self._set_pill(self._sidebar_foot_pill, f"Signed in · {email}", "success")
            else:
                self._set_pill(
                    self._sidebar_foot_pill,
                    "Not signed in — open Account to sign in",
                    "warning",
                )

    def _update_log_view(self, log_path: Path) -> None:
        if not log_path.exists():
            return
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                read_size = min(size, 8_000)
                handle.seek(size - read_size)
                tail = handle.read()
        except OSError:
            return
        lines = tail.splitlines()[-80:]
        text = "\n".join(lines)
        if text != self._log_view.toPlainText():
            self._log_view.setPlainText(text)
            scrollbar = self._log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    # --- Account page -------------------------------------------------------

    def _build_account_page(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 28, 28, 20)
        layout.setSpacing(14)

        layout.addWidget(self._build_page_heading("Account", "Sign-in and API keys"))

        account_card = QFrame()
        account_card.setStyleSheet(_ACCOUNT_CARD_QSS)
        account_layout = QVBoxLayout(account_card)
        account_layout.setContentsMargins(14, 12, 14, 14)
        account_layout.setSpacing(10)

        heading = QLabel("Sign in with ChatGPT")
        heading.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        heading.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        account_layout.addWidget(heading)

        description = QLabel(
            "Powers chat and computer-use from your ChatGPT/Codex subscription. "
            "A browser tab will open for OAuth login."
        )
        description.setFont(QFont("Segoe UI", 9))
        description.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        description.setWordWrap(True)
        account_layout.addWidget(description)

        self._account_status_label = QLabel("")
        self._account_status_label.setFont(QFont("Segoe UI", 9))
        self._account_status_label.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        self._account_status_label.setWordWrap(True)
        account_layout.addWidget(self._account_status_label)

        account_row = QHBoxLayout()
        account_row.setSpacing(8)
        account_row.addStretch()

        self._sign_in_button = QPushButton("Sign in with ChatGPT")
        self._sign_in_button.setStyleSheet(_BUTTON_QSS)
        self._sign_in_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sign_in_button.clicked.connect(self._start_sign_in)
        account_row.addWidget(self._sign_in_button)

        self._sign_out_button = QPushButton("Sign out")
        self._sign_out_button.setStyleSheet(_GHOST_BUTTON_QSS)
        self._sign_out_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sign_out_button.clicked.connect(self._sign_out)
        account_row.addWidget(self._sign_out_button)

        account_layout.addLayout(account_row)
        layout.addWidget(account_card)

        env_values = read_env(self._env_path)
        current_key = env_values.get("OPENAI_API_KEY", "")

        audio_card = QFrame()
        audio_card.setStyleSheet(_VOICE_CARD_QSS)
        audio_layout = QVBoxLayout(audio_card)
        audio_layout.setContentsMargins(18, 16, 18, 18)
        audio_layout.setSpacing(10)

        audio_heading = QLabel("OpenAI API key (optional, for voice)")
        audio_heading.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        audio_heading.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        audio_layout.addWidget(audio_heading)

        audio_description = QLabel(
            "Voice features (mic transcription + spoken replies) need this. "
            "Chat and computer-use don't."
        )
        audio_description.setFont(QFont("Segoe UI", 9))
        audio_description.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        audio_description.setWordWrap(True)
        audio_layout.addWidget(audio_description)

        self._api_key_input = QLineEdit()
        self._api_key_input.setStyleSheet(_INPUT_QSS)
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setText(current_key)
        audio_layout.addWidget(self._api_key_input)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._show_key_button = QPushButton("Show")
        self._show_key_button.setCheckable(True)
        self._show_key_button.setStyleSheet(_GHOST_BUTTON_QSS)
        self._show_key_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._show_key_button.toggled.connect(self._toggle_key_visibility)
        row.addWidget(self._show_key_button)

        row.addStretch()

        self._api_key_status = QLabel("")
        self._api_key_status.setFont(QFont("Segoe UI", 9))
        self._api_key_status.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        row.addWidget(self._api_key_status)

        save_button = QPushButton("Save")
        save_button.setStyleSheet(_GHOST_BUTTON_QSS)
        save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_button.clicked.connect(self._save_api_key)
        row.addWidget(save_button)

        save_restart_button = QPushButton("Save & restart")
        save_restart_button.setStyleSheet(_BUTTON_QSS)
        save_restart_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_restart_button.clicked.connect(self._save_api_key_and_restart)
        row.addWidget(save_restart_button)

        audio_layout.addLayout(row)
        layout.addWidget(audio_card)

        layout.addWidget(self._build_custom_api_card(env_values))
        layout.addStretch()

        self._login_thread: QThread | None = None
        self._login_worker: _LoginWorker | None = None
        self._refresh_account_card()
        return container

    def _build_custom_api_card(self, env_values: dict[str, str]) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_VOICE_CARD_QSS)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        card_layout.setSpacing(10)

        heading = QLabel("Custom API (power Helper with any OpenAI-compatible API)")
        heading.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        heading.setStyleSheet(
            f"color: {_COLOR_TEXT_PRIMARY}; background: transparent; border: none;"
        )
        card_layout.addWidget(heading)

        description = QLabel(
            "Point Helper at any OpenAI-compatible /v1/chat/completions endpoint "
            "(OpenAI, Groq, OpenRouter, Together, LM Studio, vLLM, Ollama, …). "
            "When base URL and key are both set, chat, reasoning, and computer-use route here "
            "instead of the ChatGPT sign-in. Leave blank to keep using ChatGPT."
        )
        description.setFont(QFont("Segoe UI", 9))
        description.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        description.setWordWrap(True)
        card_layout.addWidget(description)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._custom_api_base_input = QLineEdit()
        self._custom_api_base_input.setStyleSheet(_INPUT_QSS)
        self._custom_api_base_input.setPlaceholderText("https://api.openai.com/v1")
        self._custom_api_base_input.setText(env_values.get("HELPER_API_BASE_URL", ""))

        self._custom_api_key_input = QLineEdit()
        self._custom_api_key_input.setStyleSheet(_INPUT_QSS)
        self._custom_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._custom_api_key_input.setText(env_values.get("HELPER_API_KEY", ""))

        self._custom_api_model_input = QLineEdit()
        self._custom_api_model_input.setStyleSheet(_INPUT_QSS)
        self._custom_api_model_input.setPlaceholderText("gpt-4o-mini, llama-3.1-70b, …")
        self._custom_api_model_input.setText(env_values.get("HELPER_API_MODEL", ""))

        for label_text, widget in (
            ("Base URL", self._custom_api_base_input),
            ("API key", self._custom_api_key_input),
            ("Model", self._custom_api_model_input),
        ):
            label_widget = QLabel(label_text)
            label_widget.setStyleSheet(
                f"color: {_COLOR_TEXT_SECONDARY}; background: transparent; border: none;"
            )
            label_widget.setFont(QFont("Segoe UI", 9))
            form.addRow(label_widget, widget)
        card_layout.addLayout(form)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._custom_api_show_button = QPushButton("Show")
        self._custom_api_show_button.setCheckable(True)
        self._custom_api_show_button.setStyleSheet(_GHOST_BUTTON_QSS)
        self._custom_api_show_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._custom_api_show_button.toggled.connect(self._toggle_custom_api_key_visibility)
        row.addWidget(self._custom_api_show_button)

        row.addStretch()

        self._custom_api_status = QLabel("")
        self._custom_api_status.setFont(QFont("Segoe UI", 9))
        self._custom_api_status.setStyleSheet(
            f"color: {_COLOR_TEXT_MUTED}; background: transparent; border: none;"
        )
        row.addWidget(self._custom_api_status)

        save_button = QPushButton("Save")
        save_button.setStyleSheet(_GHOST_BUTTON_QSS)
        save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_button.clicked.connect(self._save_custom_api)
        row.addWidget(save_button)

        save_restart_button = QPushButton("Save & restart")
        save_restart_button.setStyleSheet(_BUTTON_QSS)
        save_restart_button.setCursor(Qt.CursorShape.PointingHandCursor)
        save_restart_button.clicked.connect(self._save_custom_api_and_restart)
        row.addWidget(save_restart_button)

        card_layout.addLayout(row)
        return card

    def _toggle_custom_api_key_visibility(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self._custom_api_key_input.setEchoMode(mode)
        self._custom_api_show_button.setText("Hide" if visible else "Show")

    def _save_custom_api(self) -> bool:
        base = self._custom_api_base_input.text().strip().rstrip("/")
        key = self._custom_api_key_input.text().strip()
        model = self._custom_api_model_input.text().strip()
        if (base or key) and not (base and key):
            self._custom_api_status.setText("Both Base URL and API key are required.")
            return False
        updates = {
            "HELPER_API_BASE_URL": base,
            "HELPER_API_KEY": key,
            "HELPER_API_MODEL": model,
        }
        try:
            write_env(self._env_path, updates)
        except Exception as exc:
            log.exception("Saving custom API config failed")
            self._custom_api_status.setText(f"Save failed: {exc}")
            return False
        self._custom_api_status.setText(
            "Saved. Restart to apply." if base and key else "Cleared — using ChatGPT sign-in."
        )
        return True

    def _save_custom_api_and_restart(self) -> None:
        if self._save_custom_api():
            self._restart_callback()

    def _refresh_account_card(self) -> None:
        if not hasattr(self, "_sign_in_button"):
            return
        signed_in = oauth_codex.is_signed_in()
        if signed_in:
            email = oauth_codex.account_email() or "unknown account"
            self._account_status_label.setText(f"Signed in as <b>{email}</b>")
            self._sign_in_button.setText("Re-authenticate")
            self._sign_out_button.setVisible(True)
        else:
            self._account_status_label.setText("Not signed in.")
            self._sign_in_button.setText("Sign in with ChatGPT")
            self._sign_out_button.setVisible(False)

    def _start_sign_in(self) -> None:
        if self._login_thread is not None:
            return
        self._sign_in_button.setEnabled(False)
        self._account_status_label.setText("Waiting for browser sign-in…")

        self._login_thread = QThread(self)
        self._login_worker = _LoginWorker()
        self._login_worker.moveToThread(self._login_thread)
        self._login_thread.started.connect(self._login_worker.run)
        self._login_worker.finished.connect(self._on_sign_in_finished)
        self._login_worker.finished.connect(self._login_thread.quit)
        self._login_worker.finished.connect(self._login_worker.deleteLater)
        self._login_thread.finished.connect(self._on_login_thread_finished)
        self._login_thread.start()

    def _on_sign_in_finished(self, ok: bool, message: str) -> None:
        if ok:
            self._account_status_label.setText("Signed in. Helper is ready.")
            if self._auth_changed_callback is not None:
                self._auth_changed_callback(True)
        else:
            self._account_status_label.setText(f"Sign-in failed: {message}")
        self._sign_in_button.setEnabled(True)
        self._refresh_account_card()

    def _on_login_thread_finished(self) -> None:
        if self._login_thread is not None:
            self._login_thread.deleteLater()
        self._login_thread = None
        self._login_worker = None

    def _sign_out(self) -> None:
        oauth_codex.sign_out()
        self._refresh_account_card()
        self._account_status_label.setText("Signed out.")
        if self._auth_changed_callback is not None:
            self._auth_changed_callback(False)

    def _toggle_key_visibility(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self._api_key_input.setEchoMode(mode)
        self._show_key_button.setText("Hide" if visible else "Show")

    def _save_api_key(self) -> bool:
        value = self._api_key_input.text().strip()
        try:
            write_env(self._env_path, {"OPENAI_API_KEY": value})
        except Exception as exc:
            log.exception("Saving API key failed")
            self._api_key_status.setText(f"Save failed: {exc}")
            return False
        self._api_key_status.setText("Saved. Restart to apply.")
        return True

    def _save_api_key_and_restart(self) -> None:
        if self._save_api_key():
            self._restart_callback()

    # --- Window lifecycle ---------------------------------------------------

    def show_dashboard(self) -> None:
        self._reload_from_env()
        self.show()
        self.raise_()
        self.activateWindow()
        self._refresh_status()
        self._refresh_conversations(force=True)
        if not self._status_timer.isActive():
            self._status_timer.start()

    def _reload_from_env(self) -> None:
        env_values = read_env(self._env_path)
        for env_key, widget in self._field_widgets.items():
            value = _env_value(env_values, env_key, "")
            if value == "":
                # Boolean/toggle fields legitimately serialize as "false";
                # don't skip them just because they're falsy.
                if widget.property("kind") not in ("toggle",):
                    continue
            kind = widget.property("kind") if widget is not None else None
            if kind == "slider_float":
                spin = widget.property("spin")
                if isinstance(spin, QDoubleSpinBox):
                    try:
                        spin.setValue(float(value))
                    except ValueError:
                        pass
            elif kind == "toggle":
                assert isinstance(widget, QCheckBox)
                widget.setChecked(str(value).strip().lower() in {"1", "true", "yes", "on"})
            elif kind == "provider":
                assert isinstance(widget, QComboBox)
                target = (value or "codex").strip().lower()
                for i in range(widget.count()):
                    if widget.itemData(i) == target:
                        widget.setCurrentIndex(i)
                        break
            elif kind == "model":
                assert isinstance(widget, QComboBox)
                widget.setEditText(value)
            elif kind == "theme":
                group = widget.property("group")
                if group is not None:
                    for btn in group.buttons():
                        if btn.property("themeKey") == value:
                            btn.setChecked(True)
                            break
            elif isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QSpinBox):
                try:
                    widget.setValue(int(value))
                except ValueError:
                    pass
            elif isinstance(widget, QDoubleSpinBox):
                try:
                    widget.setValue(float(value))
                except ValueError:
                    pass
            elif isinstance(widget, QComboBox):
                idx = widget.findText(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            self._original_values[env_key] = self._widget_value(widget)
        if hasattr(self, "_field_widgets") and self._field_widgets:
            self._refresh_provider_dependents()
        if hasattr(self, "_api_key_input"):
            self._api_key_input.setText(env_values.get("OPENAI_API_KEY", ""))
        if hasattr(self, "_custom_api_base_input"):
            self._custom_api_base_input.setText(env_values.get("HELPER_API_BASE_URL", ""))
            self._custom_api_key_input.setText(env_values.get("HELPER_API_KEY", ""))
            self._custom_api_model_input.setText(env_values.get("HELPER_API_MODEL", ""))
        if hasattr(self, "_save_status"):
            self._set_pill(self._save_status, "", "neutral")
        if hasattr(self, "_api_key_status"):
            self._api_key_status.setText("")
        if hasattr(self, "_custom_api_status"):
            self._custom_api_status.setText("")
        self._refresh_account_card()

    def hide_window(self) -> None:
        self._status_timer.stop()
        self.hide()
        self.closed.emit()

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide_window()
        else:
            self.show_dashboard()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.hide_window()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._status_timer.stop()
        event.accept()
        self.closed.emit()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._frame.resize(self.size())


class _LoginWorker(QObject):
    finished = pyqtSignal(bool, str)

    def run(self) -> None:
        try:
            handle = oauth_codex.start_login()
            handle.wait(timeout=180)
            self.finished.emit(True, "")
        except Exception as exc:
            log.exception("ChatGPT sign-in failed")
            self.finished.emit(False, str(exc))
