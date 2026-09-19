from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from typing import TYPE_CHECKING

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon
import re

from config import (
    DEFAULT_MODE,
    ENV_PATH,
    HOTKEY,
    LOOP_SETTLE_SEC,
    ROOT,
    SPEAK_TYPED_CHAT,
    STARTUP_LOG,
    setup_logging,
)
from computer_control import sleep_with_abort
from conversation_store import ConversationStore, StoredConversation
from help_diagnostics import HelpTargetDiagnosticSink
from launch_safety import LaunchError, launch_target, safe_target_label
import oauth_codex
from ui_chat import ChatWindow
from ui_dashboard import DashboardWindow
from ui_ghost_cursor import GhostCursorManager
from ui_overlay import OverlayManager
from ui_widget import ChatStatusPill, FloatingCircle
from windows_integration import configure_windows_app_identity, ensure_windows_shortcuts

log = logging.getLogger("helper")

if TYPE_CHECKING:
    from agent import GuideAction, GuideSession, GuideTurn, HelplerAgent
    from audio_handler import AudioHandler
    from computer_control import ConfirmationRequest, ComputerController
    from help_session import HelpSession


ALIASES = {
    "control": "ctrl",
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "shift_l": "shift",
    "shift_r": "shift",
    "alt_l": "alt",
    "alt_r": "alt",
    "cmd": "meta",
    "cmd_l": "meta",
    "cmd_r": "meta",
    " ": "space",
}

CHAT_TAP_THRESHOLD_SEC = 0.3
CONFIRMATION_TIMEOUT_SEC = 120.0
APP_ICON_PATH = ROOT / "assets" / "helper_logo.ico"
CONVERSATIONS_PATH = ROOT / "data" / "conversations.json"

INTENT_WITHOUT_ACTION_RE = re.compile(
    r"^\s*(i'll|i will|let me|i'm going to|going to|sure[,!.]?\s+i|on it[,!.]?\s+i)",
    re.IGNORECASE,
)
ZERO_ACTION_NUDGE = (
    "You said you'd act but did not call a tool. Call the appropriate tool now to perform the requested action."
)


def run_active_continuation_loop(
    agent,  # HelplerAgent
    session,  # GuideSession
    abort,  # AbortController
    initial_turn,  # GuideTurn
    settle_sec: float,
    on_turn=lambda turn: None,
    nudge_message: str = ZERO_ACTION_NUDGE,
    intent_pattern=INTENT_WITHOUT_ACTION_RE,
    sleep_fn=sleep_with_abort,
    max_verifier_rounds: int = 1,
    max_consecutive_nudges: int = 1,
    provider_retry_backoff: tuple[float, ...] = (0.5, 1.0, 2.0),
):
    """Drive the active-mode continuation loop until the goal is complete or the model stops.

    Reliability hooks (see plans/modular-tumbling-lemur.md):
    - mid-loop intent nudge with consecutive cap so "I'll do X" without a tool call
      doesn't silently end the run, but can't infinite-loop either;
    - goal-completion verifier before exiting on a zero-action turn (only when the
      session actually performed actions); capped at `max_verifier_rounds`;
    - graceful wrap-up turn on `LimitExceeded` or exhausted provider retries;
    - bounded provider-error retry around each `continue_guide` call.

    Pure of UI concerns: `on_turn` is invoked once per turn produced (excluding the
    initial turn passed in), and `sleep_fn(abort, settle_sec)` paces the iterations.
    """
    from agent import GuideTurn, HelplerAgent, LimitExceeded
    try:
        from openai_client import (
            BadProviderResponse,
            ProviderUnavailable,
            RateLimited,
        )
        transient_provider_errors: tuple[type[BaseException], ...] = (
            RateLimited,
            ProviderUnavailable,
            BadProviderResponse,
        )
    except Exception:
        transient_provider_errors = ()

    turn = initial_turn
    saw_any_action = bool(initial_turn.actions)
    consecutive_nudges = 0
    verifier_rounds = 0

    def _step_with_retry(note=None):
        backoffs = list(provider_retry_backoff)
        attempt = 0
        last_exc: BaseException | None = None
        while True:
            try:
                if note is None:
                    return agent.continue_guide(session)
                return agent.continue_guide(session, note=note)
            except transient_provider_errors as exc:
                last_exc = exc
                if attempt >= len(backoffs) or abort.is_aborted():
                    raise
                log.warning("Transient provider error (attempt %d): %s", attempt + 1, exc)
                try:
                    sleep_fn(abort, backoffs[attempt])
                except Exception:
                    raise last_exc from None
                if abort.is_aborted():
                    raise last_exc from None
                attempt += 1

    def _emit_wrapup(message: str) -> "GuideTurn":
        wrap = GuideTurn(
            message=message,
            actions=[],
            done=True,
            step_index=getattr(session, "step_count", 0),
            elapsed_sec=0.0,
            capture=None,
        )
        on_turn(wrap)
        return wrap

    def _wrapup_for_limit(exc: "LimitExceeded") -> "GuideTurn":
        kind = getattr(exc, "kind", "")
        if kind == "steps":
            reason = "I've reached the maximum step count for this task."
        elif kind == "timeout":
            reason = "I've reached the time limit for this task."
        else:
            reason = "I've reached a limit for this task."
        executed = session.history.last_executed_actions() if hasattr(session, "history") else []
        summaries = [
            HelplerAgent._summarize_action(item["name"], item.get("args") or {})
            for item in executed[-3:]
            if item.get("name")
        ]
        joined = " ".join(s.rstrip(".") + "." for s in summaries if s)
        tail = f" Last actions: {joined}" if joined else ""
        return _emit_wrapup(f"{reason}{tail} Ask me to continue if you want me to keep going.")

    def _wrapup_for_provider(exc: BaseException) -> "GuideTurn":
        log.warning("Provider retries exhausted; stopping loop: %s", exc)
        return _emit_wrapup(
            "I hit a transient provider error and stopped. Try again in a moment."
        )

    # Initial-turn intent nudge: model said "I'll do X" but emitted no tool calls.
    if (
        not turn.actions
        and not abort.is_aborted()
        and intent_pattern.search(turn.message or "")
        and consecutive_nudges < max_consecutive_nudges
    ):
        log.info("Initial turn had intent without a tool call; nudging.")
        try:
            turn = _step_with_retry(note=nudge_message)
        except LimitExceeded as exc:
            return _wrapup_for_limit(exc)
        except transient_provider_errors as exc:
            return _wrapup_for_provider(exc)
        consecutive_nudges = 1
        saw_any_action = saw_any_action or bool(turn.actions)
        on_turn(turn)

    while not abort.is_aborted():
        if turn.actions:
            try:
                sleep_fn(abort, settle_sec)
            except Exception:
                break
            if abort.is_aborted():
                break
            try:
                turn = _step_with_retry()
            except LimitExceeded as exc:
                return _wrapup_for_limit(exc)
            except transient_provider_errors as exc:
                return _wrapup_for_provider(exc)
            consecutive_nudges = 0
            saw_any_action = saw_any_action or bool(turn.actions)
            on_turn(turn)
            continue

        # Zero-action turn — decide between nudging, verifying, or exiting.
        if (
            intent_pattern.search(turn.message or "")
            and consecutive_nudges < max_consecutive_nudges
        ):
            log.info("Mid-loop intent without action; nudging.")
            try:
                sleep_fn(abort, settle_sec)
            except Exception:
                break
            if abort.is_aborted():
                break
            try:
                turn = _step_with_retry(note=nudge_message)
            except LimitExceeded as exc:
                return _wrapup_for_limit(exc)
            except transient_provider_errors as exc:
                return _wrapup_for_provider(exc)
            consecutive_nudges += 1
            saw_any_action = saw_any_action or bool(turn.actions)
            on_turn(turn)
            continue

        verifier = getattr(agent, "verify_goal_complete", None)
        if (
            saw_any_action
            and verifier is not None
            and verifier_rounds < max_verifier_rounds
        ):
            verifier_rounds += 1
            try:
                done, note = verifier(session)
            except Exception as exc:
                log.warning("Goal verifier failed; exiting loop: %s", exc)
                break
            if done:
                log.info("Goal verifier confirmed completion: %s", note)
                break
            log.info("Goal verifier rejected exit: %s", note)
            followup = (
                f"Goal verifier says the task is not done: {note}. "
                "Continue working on the original goal."
            )
            try:
                sleep_fn(abort, settle_sec)
            except Exception:
                break
            if abort.is_aborted():
                break
            try:
                turn = _step_with_retry(note=followup)
            except LimitExceeded as exc:
                return _wrapup_for_limit(exc)
            except transient_provider_errors as exc:
                return _wrapup_for_provider(exc)
            consecutive_nudges = 0
            saw_any_action = saw_any_action or bool(turn.actions)
            on_turn(turn)
            continue

        break

    return turn


def helper_app_icon(app: QApplication) -> QIcon:
    icon = QIcon(str(APP_ICON_PATH))
    if not icon.isNull():
        return icon
    return app.style().standardIcon(app.style().StandardPixmap.SP_ComputerIcon)


def normalize_key_name(name: str) -> str:
    return ALIASES.get(name.lower(), name.lower())


def key_name(key: object) -> str | None:
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        return normalize_key_name(char)

    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return normalize_key_name(name)

    return None


@dataclass
class PendingConfirmation:
    request: ConfirmationRequest
    event: Event = field(default_factory=Event)
    approved: bool = False
    cancelled: bool = False


class ConfirmationBroker(QObject):
    confirmation_requested = pyqtSignal(object)

    def __init__(self, parent_widget: FloatingCircle) -> None:
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._gui_thread_id = get_ident()
        self._pending_lock = Lock()
        self._pending: dict[int, PendingConfirmation] = {}
        self._closed = Event()
        self.confirmation_requested.connect(self._resolve_pending)

    def confirm(self, request: ConfirmationRequest) -> bool:
        if self._closed.is_set():
            return False
        pending = PendingConfirmation(request=request)
        if get_ident() == self._gui_thread_id:
            self._apply_confirmation(pending)
        else:
            pending_key = id(pending)
            with self._pending_lock:
                self._pending[pending_key] = pending
            try:
                self.confirmation_requested.emit(pending)
                if not pending.event.wait(CONFIRMATION_TIMEOUT_SEC):
                    pending.cancelled = True
                    pending.event.set()
                    log.warning("Confirmation timed out; blocking action: %s", request.action)
                    return False
            finally:
                with self._pending_lock:
                    self._pending.pop(pending_key, None)
        return pending.approved

    def _resolve_pending(self, pending: PendingConfirmation) -> None:
        if self._closed.is_set() or pending.cancelled:
            pending.approved = False
            pending.event.set()
            return
        self._apply_confirmation(pending)

    def _apply_confirmation(self, pending: PendingConfirmation) -> None:
        if pending.cancelled:
            pending.event.set()
            return
        pending.approved = self._show_dialog(pending.request)
        pending.event.set()

    def _show_dialog(self, request: ConfirmationRequest) -> bool:
        box = QMessageBox(self._parent_widget)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Confirm Helper Action")
        box.setText(request.reason)

        details = [request.summary]
        if request.foreground_window is not None:
            title = request.foreground_window.title or "<untitled>"
            class_name = request.foreground_window.class_name or "<unknown class>"
            details.append(f"Window: {title}")
            details.append(f"Class: {class_name}")

        box.setInformativeText("\n".join(details))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == int(QMessageBox.StandardButton.Yes)

    def shutdown(self) -> None:
        self._closed.set()
        with self._pending_lock:
            pending = list(self._pending.values())
        for item in pending:
            item.cancelled = True
            item.approved = False
            item.event.set()


HELP_BLOCKED_ACTIONS = frozenset(
    {
        "click_at",
        "double_click_at",
        "right_click_at",
        "click_control",
        "type_text_at",
        "drag_and_drop",
    }
)


class DesktopActionDispatcher:
    def __init__(self, controller: ComputerController) -> None:
        self._controller = controller

    def dispatch(self, action: GuideAction) -> dict[str, object]:
        log.info("Action: %s", action.summary)

        if (
            self._controller.mode == FloatingCircle.HELP
            and action.name in HELP_BLOCKED_ACTIONS
        ):
            log.info("Blocked in Help mode: %s", action.name)
            return self._blocked_result(
                action,
                "Clicks and typing are not performed in Help mode. Switch to Active mode if you want Helper to do it for you.",
            )

        if action.name == "launch_app":
            command = str(action.raw_args.get("command") or "").strip()
            display_name = str(action.raw_args.get("display_name") or command or "app")
            if not command:
                return self._blocked_result(action, "No launch command was provided.")
            return self._launch_target(action, command, f"Open {display_name}.")

        if action.name == "open_web_browser":
            return self._launch_target(action, "", action.summary or "Open the browser.")

        if action.name == "navigate":
            url = str(action.raw_args.get("url") or "").strip()
            if not url:
                return self._blocked_result(action, "No URL was provided.")
            return self._launch_target(action, url, f"Navigate to {safe_target_label(url)}.")

        if action.name == "click_control":
            label = str(action.raw_args.get("label") or "").strip()
            if not label:
                return self._blocked_result(action, "No UI control label was provided.")
            return self._click_control(action, label)

        if action.name == "click_at" and self._has_point(action):
            return self._report(
                self._controller.click(
                    action.screen_x,
                    action.screen_y,
                    description=action.summary,
                    **self._confirmation_kwargs(action),
                )
            )

        if action.name == "hover_at" and self._has_point(action):
            return self._report(
                self._controller.move_to(
                    action.screen_x,
                    action.screen_y,
                    description=action.summary,
                    **self._confirmation_kwargs(action),
                )
            )

        if action.name == "type_text_at" and self._has_point(action):
            results = []
            results.append(self._report(
                self._controller.click(
                    action.screen_x,
                    action.screen_y,
                    description="Focus the target input.",
                )
            ))
            if action.clear_before_typing:
                results.append(self._report(
                    self._controller.key(
                        "ctrl+a",
                        description="Select existing text.",
                    )
                ))
                results.append(self._report(
                    self._controller.key(
                        "backspace",
                        description="Clear existing text.",
                    )
                ))

            confirmation_kwargs: dict[str, object] = {}
            if action.requires_confirmation and not action.press_enter:
                confirmation_kwargs = self._confirmation_kwargs(action)

            if action.text:
                results.append(self._report(
                    self._controller.type_text(
                        action.text,
                        description=action.summary,
                        **confirmation_kwargs,
                    )
                ))
            if action.press_enter:
                results.append(self._report(
                    self._controller.key(
                        "enter",
                        description=f"Submit typed text. {action.summary}",
                        **self._confirmation_kwargs(action),
                    )
                ))
            return {"status": "ok", "results": results}

        if action.name == "key_combination" and action.keys:
            return self._report(
                self._controller.key(
                    action.keys,
                    description=action.summary,
                    force_execute=bool(action.raw_args.get("fast_local")),
                    **self._confirmation_kwargs(action),
                )
            )

        if action.name == "scroll_document":
            delta = self._scroll_delta(action.direction)
            if delta is None:
                return self._blocked_result(action, f"Horizontal scroll is not supported: {action.direction}")
            return self._report(
                self._controller.scroll(
                    delta,
                    description=action.summary,
                    **self._confirmation_kwargs(action),
                )
            )

        if action.name == "scroll_at" and self._has_point(action):
            delta = self._scroll_delta(action.direction)
            if delta is None:
                return self._blocked_result(action, f"Horizontal scroll is not supported: {action.direction}")
            return self._report(
                self._controller.scroll(
                    delta,
                    x=action.screen_x,
                    y=action.screen_y,
                    description=action.summary,
                    **self._confirmation_kwargs(action),
                )
            )

        if (
            action.name == "drag_and_drop"
            and self._has_point(action)
            and action.destination_screen_x is not None
            and action.destination_screen_y is not None
        ):
            results = []
            results.append(self._report(
                self._controller.move_to(
                    action.screen_x,
                    action.screen_y,
                    description="Move to the drag starting point.",
                )
            ))
            results.append(self._report(
                self._controller.drag_to(
                    action.destination_screen_x,
                    action.destination_screen_y,
                    description=action.summary,
                    **self._confirmation_kwargs(action),
                )
            ))
            return {"status": "ok", "results": results}

        if action.name == "wait_5_seconds":
            self._sleep_with_abort(5.0)
            log.info("Waited 5 seconds.")
            return {"status": "ok", "action": "wait_5_seconds", "executed": True}

        log.info("Skipping unsupported action: %s", action.name)
        return self._blocked_result(action, f"Unsupported action: {action.name}")

    def _launch_target(self, action: GuideAction, target: str, description: str) -> dict[str, object]:
        launch_value = target or "about:blank"
        self._controller.abort_controller.checkpoint()
        try:
            launch_target(launch_value)
        except (LaunchError, OSError, ValueError) as exc:
            log.exception("Launch failed: %s", description)
            return {
                "status": "blocked",
                "action": action.name,
                "target": launch_value,
                "executed": False,
                "blocked_reason": str(exc),
            }

        log.info("Executed: %s", description)
        return {
            "status": "executed",
            "action": action.name,
            "target": launch_value,
            "executed": True,
            "mode": self._controller.mode,
        }

    def _click_control(self, action: GuideAction, label: str) -> dict[str, object]:
        self._controller.abort_controller.checkpoint()
        try:
            from pywinauto import Desktop
        except Exception as exc:
            return self._blocked_result(
                action,
                f"Windows UI Automation is unavailable: {exc}",
            )

        normalized = label.lower()
        try:
            desktop = Desktop(backend="uia")
            roots = []
            try:
                roots.append(desktop.get_active())
            except Exception:
                pass
            roots.extend(desktop.windows(visible_only=True, enabled_only=True))

            seen: set[int] = set()
            for root in roots:
                handle = getattr(getattr(root, "element_info", None), "handle", id(root))
                if handle in seen:
                    continue
                seen.add(handle)
                candidates = [root]
                try:
                    candidates.extend(root.descendants())
                except Exception:
                    continue
                for control in candidates:
                    try:
                        text = (control.window_text() or "").strip()
                        automation_id = (control.element_info.automation_id or "").strip()
                        control_type = (control.element_info.control_type or "").strip().lower()
                    except Exception:
                        continue
                    haystack = " ".join(part.lower() for part in (text, automation_id) if part)
                    if not haystack or normalized not in haystack:
                        continue
                    if control_type and control_type not in {
                        "button",
                        "menuitem",
                        "tabitem",
                        "hyperlink",
                        "listitem",
                        "treeitem",
                        "text",
                        "edit",
                    }:
                        continue
                    try:
                        control.click_input()
                    except Exception:
                        control.set_focus()
                        control.click_input()
                    log.info("Executed UIA click: %s", label)
                    return {
                        "status": "executed",
                        "action": action.name,
                        "label": label,
                        "executed": True,
                        "mode": self._controller.mode,
                        "control_text": text,
                        "control_type": control_type,
                    }
        except Exception as exc:
            log.exception("UIA click failed: %s", label)
            return self._blocked_result(action, f"UI Automation click failed: {exc}")

        return self._blocked_result(action, f"No visible UI control matched: {label}")

    @staticmethod
    def _has_point(action: GuideAction) -> bool:
        return action.screen_x is not None and action.screen_y is not None

    @staticmethod
    def _scroll_delta(direction: str | None) -> int | None:
        normalized = (direction or "down").lower()
        if normalized == "down":
            return -700
        if normalized == "up":
            return 700
        return None

    @staticmethod
    def _confirmation_kwargs(action: GuideAction) -> dict[str, object]:
        if not action.requires_confirmation:
            return {}
        return {
            "requires_confirmation": True,
            "confirmation_reason": action.confirmation_reason
            or "Model requested confirmation.",
        }

    @staticmethod
    def _report(result) -> dict[str, object]:
        payload = asdict(result)
        if result.confirmation_required and result.confirmation_granted is False:
            log.info("Blocked: %s", result.blocked_reason)
            payload["status"] = "blocked"
        elif result.executed:
            log.info("Executed: %s", result.action)
            payload["status"] = "executed"
        else:
            log.info("Guided: %s", result.action)
            payload["status"] = "guided"
        return payload

    @staticmethod
    def _blocked_result(action: GuideAction, reason: str) -> dict[str, object]:
        return {
            "status": "blocked",
            "action": action.name,
            "executed": False,
            "blocked_reason": reason,
        }

    def _sleep_with_abort(self, duration_sec: float) -> None:
        deadline = time.perf_counter() + max(duration_sec, 0.0)
        while True:
            self._controller.abort_controller.checkpoint()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))


class HelplerDesktopApp(QObject):
    widget_state_requested = pyqtSignal(str)
    caption_changed = pyqtSignal(str)
    overlay_highlight_requested = pyqtSignal(
        int, int, int, int, str, int, int, int
    )
    overlay_clear_requested = pyqtSignal()
    help_overlay_clear_sync_requested = pyqtSignal(object)
    agent_message_received = pyqtSignal(str)
    user_message_received = pyqtSignal(str)
    chat_toggle_requested = pyqtSignal()
    chat_show_requested = pyqtSignal()
    chat_status_changed = pyqtSignal(str)

    def __init__(self, app: QApplication, hotkey: str, startup_surface: str = "background") -> None:
        super().__init__()
        if not hotkey.strip():
            raise ValueError("HELPER_HOTKEY cannot be empty.")

        self._app = app
        self._startup_surface = startup_surface
        self._hotkey_label = hotkey
        self._hotkey = {normalize_key_name(part) for part in hotkey.split("+") if part}
        self._widget = FloatingCircle()
        self._widget.set_state(FloatingCircle.IDLE)
        self._app_icon = helper_app_icon(self._app)
        self._app.setWindowIcon(self._app_icon)
        shortcuts = ensure_windows_shortcuts()
        if shortcuts:
            log.info("Helper shortcuts refreshed: %s", ", ".join(str(path) for path in shortcuts))
        self._backend_ready = False
        self._backend_initializing = False
        self._backend_error: str | None = None
        if DEFAULT_MODE == FloatingCircle.ACTIVE:
            self._selected_mode = FloatingCircle.ACTIVE
        else:
            self._selected_mode = FloatingCircle.HELP
        self._chat_window = ChatWindow()
        self._chat_window.set_mode(self._selected_mode)
        self._chat_window.setWindowIcon(self._app_icon)
        self._status_pill = ChatStatusPill()
        self._conversation_store = ConversationStore(CONVERSATIONS_PATH)
        self._help_target_diagnostics = HelpTargetDiagnosticSink()
        self._active_conversation: StoredConversation | None = None
        self._dashboard = DashboardWindow(
            env_path=ENV_PATH,
            status_provider=self._status_snapshot,
            restart_callback=self._handle_restart_requested,
            auth_changed_callback=self._handle_auth_changed,
            conversations_provider=self._conversations_snapshot,
        )
        self._dashboard.setWindowIcon(self._app_icon)
        self._overlay = OverlayManager(app)
        self._ghost_cursor = GhostCursorManager(app)
        self.help_overlay_clear_sync_requested.connect(self._on_help_overlay_clear_sync_requested)
        self._confirmation_broker = ConfirmationBroker(self._widget)
        self._audio: AudioHandler | None = None
        self._controller: ComputerController | None = None
        self._dispatcher: DesktopActionDispatcher | None = None
        self._agent: HelplerAgent | None = None
        self._agent_error: str | None = None
        self._help_session: "HelpSession | None" = None
        self._listener = None
        self._mouse_listener = None
        self._abort_requested_type: type[BaseException] | None = None
        self._lock = Lock()
        self._pressed: set[str] = set()
        self._recording_active = False
        self._recording_started_at = 0.0
        self._busy = False
        self._shutting_down = False
        self._guide_session: GuideSession | None = None
        self._caption_thread: Thread | None = None

        self.widget_state_requested.connect(self._widget.set_state)
        self.caption_changed.connect(self._widget.set_caption)
        self.overlay_highlight_requested.connect(self._overlay.show_highlight)
        self.overlay_clear_requested.connect(self._overlay.clear)
        self.agent_message_received.connect(self._handle_agent_message)
        self.user_message_received.connect(self._handle_user_message)
        self.chat_toggle_requested.connect(self._toggle_widget_chat)
        self.chat_show_requested.connect(self._show_chat_window)
        self.chat_status_changed.connect(self._on_chat_status_changed)

        self._widget.mode_toggle_requested.connect(self._handle_mode_toggle_requested)
        self._widget.mode_select_requested.connect(self._handle_mode_select_requested)
        self._widget.quit_requested.connect(self._handle_quit_requested)
        self._widget.restart_requested.connect(self._handle_restart_requested)
        self._widget.chat_requested.connect(self._toggle_widget_chat)
        self._widget.chat_submitted.connect(self._handle_chat_submitted)
        self._widget.voice_hold_started.connect(self._start_voice_recording)
        self._widget.voice_hold_finished.connect(self._stop_voice_recording)
        self._widget.chat_open_changed.connect(self._on_widget_chat_open_changed)
        self._widget.dashboard_requested.connect(self._toggle_dashboard)
        self._widget.moved.connect(self._on_widget_moved)

        self._chat_window.submitted.connect(self._handle_chat_submitted)
        self._chat_window.closed.connect(self._on_chat_closed)
        self._chat_window.close_requested.connect(self._handle_main_window_close_requested)
        self._app.aboutToQuit.connect(self.shutdown)

        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_toggle_action = None
        self._setup_tray()

        self._widget.move(*self._initial_widget_position())
        self._widget.set_mode(self._selected_mode)
        self._widget.show()
        if self._startup_surface == "chat":
            QTimer.singleShot(0, self._show_chat_window)
        elif self._startup_surface == "dashboard":
            QTimer.singleShot(0, self._show_dashboard_window)
        self._sync_tray_actions()
        self._widget.set_state(FloatingCircle.LISTENING)
        QTimer.singleShot(2000, lambda: self._widget.set_state(FloatingCircle.IDLE))
        QTimer.singleShot(800, self._show_first_launch_balloon)
        QTimer.singleShot(50, self._initialize_backend)

    def start(self) -> None:
        log.info("Starting Helper. Loading backend for %s...", self._hotkey_label)
        log.info("Right-click the widget or tray icon for controls.")

    def _initial_widget_position(self) -> tuple[int, int]:
        screen = self._app.primaryScreen()
        if screen is None:
            screens = self._app.screens()
            screen = screens[0] if screens else None
        if screen is None:
            return (80, 80)
        geom = screen.availableGeometry()
        width, height = self._widget.width(), self._widget.height()
        x = geom.left() + (geom.width() - width) // 2
        y = geom.bottom() - height
        x = max(geom.left(), min(x, geom.right() - width))
        y = max(geom.top(), min(y, geom.bottom() - height))
        return (x, y)

    def _show_first_launch_balloon(self) -> None:
        if self._tray_icon is None:
            return
        self._tray_icon.showMessage(
            "Helper is running",
            "Look at the bottom-center of your screen. Right-click the circle for controls.",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )

    def _initialize_backend(self) -> None:
        if self._backend_ready or self._backend_initializing or self._shutting_down:
            return

        self._backend_initializing = True
        try:
            from pynput import keyboard, mouse

            from audio_handler import AudioHandler
            from computer_control import AbortRequested, ComputerController

            audio = AudioHandler()
            controller = ComputerController(
                mode=self._selected_mode,
                overlay_callback=self._handle_overlay_action,
                confirmation_callback=self._confirmation_broker.confirm,
            )
            dispatcher = DesktopActionDispatcher(controller)
            listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            mouse_listener = mouse.Listener(on_click=self._on_mouse_click)

            self._audio = audio
            self._controller = controller
            self._dispatcher = dispatcher
            self._agent = None
            self._listener = listener
            self._mouse_listener = mouse_listener
            self._abort_requested_type = AbortRequested

            if self._shutting_down:
                return

            self._listener.start()
            self._mouse_listener.start()
            self._backend_ready = True
            self._backend_error = None
            self._ensure_agent(show_errors=False)
            self.widget_state_requested.emit(FloatingCircle.IDLE)
            if self._agent is None:
                log.info(
                    "Helper shell ready in %s mode. Sign in with ChatGPT to enable chat and computer-use.",
                    self._selected_mode.upper(),
                )
            else:
                log.info(
                    "Helper ready in %s mode. Tap %s to toggle chat or hold it to record.",
                    self._selected_mode.upper(),
                    self._hotkey_label,
                )
        except Exception as exc:
            log.exception("Backend init failed")
            self._audio = None
            self._controller = None
            self._dispatcher = None
            self._agent = None
            self._agent_error = None
            self._listener = None
            self._mouse_listener = None
            self._abort_requested_type = None
            self._backend_ready = False
            self._backend_error = str(exc)
            self.caption_changed.emit(f"Startup failed: {str(exc)[:60]}")
            self.widget_state_requested.emit(FloatingCircle.IDLE)
            if self._tray_icon is not None:
                self._tray_icon.showMessage(
                    "Helper startup failed",
                    str(exc)[:200],
                    QSystemTrayIcon.MessageIcon.Critical,
                    8000,
                )
        finally:
            self._backend_initializing = False

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._confirmation_broker.shutdown()
        self._persist_active_conversation()
        if self._help_session is not None:
            self._help_session.cancel()
        if self._controller is not None:
            self._controller.abort_controller.request_abort("Helper is shutting down.")
        if self._listener is not None:
            self._listener.stop()
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
        if self._audio is not None:
            self._audio.cancel_recording()
            self._audio.stop_playback()
        self._chat_window.hide()
        self._status_pill.hide()
        self._dashboard.hide()
        self._overlay.clear()
        self._ghost_cursor.clear()
        self._widget.hide()
        self._status_pill.hide()
        if self._tray_icon is not None:
            self._tray_icon.hide()

    def _persist_active_conversation(self) -> None:
        conversation = self._active_conversation
        if conversation is None or not conversation.has_user_message():
            return
        conversation.title = conversation.derive_title()
        if conversation.ended_at < conversation.started_at:
            conversation.ended_at = time.time()
        try:
            self._conversation_store.save(conversation)
            log.info("Saved conversation %s (%d messages)", conversation.id, len(conversation.messages))
        except Exception:
            log.exception("Saving conversation failed")
        self._active_conversation = None

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            log.warning("System tray unavailable")
            QMessageBox.warning(
                None,
                "Helper",
                "System tray is not available. The floating circle is still on screen - "
                "right-click it for controls.",
            )
            return

        tray_icon = QSystemTrayIcon(self._app)
        tray_icon.setToolTip("Helper")
        tray_icon.setIcon(self._app_icon)

        tray_menu = QMenu()
        self._tray_toggle_action = tray_menu.addAction("Show Helper")
        self._tray_toggle_action.triggered.connect(self._toggle_chat_window)

        dashboard_action = tray_menu.addAction("Dashboard")
        dashboard_action.triggered.connect(self._toggle_dashboard)

        tray_menu.addSeparator()

        restart_action = tray_menu.addAction("Restart")
        restart_action.triggered.connect(self._handle_restart_requested)

        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self._handle_quit_requested)

        tray_icon.setContextMenu(tray_menu)
        tray_icon.activated.connect(self._handle_tray_activated)
        tray_icon.show()

        self._tray_icon = tray_icon
        self._sync_tray_actions()

    def _position_chat_window(self) -> None:
        return

    def _sync_tray_actions(self) -> None:
        if self._tray_toggle_action is None:
            return
        self._tray_toggle_action.setText(
            "Minimize Helper"
            if self._chat_window.isVisible() and not self._chat_window.isMinimized()
            else "Show Helper"
        )

    def _toggle_chat_window(self) -> None:
        self._chat_window.toggle_visibility()
        self._sync_tray_actions()

    def _show_chat_window(self) -> None:
        self._chat_window.show_chat()
        self._sync_tray_actions()

    def _show_dashboard_window(self) -> None:
        self._dashboard.show_dashboard()
        self._sync_tray_actions()

    def _toggle_widget_chat(self) -> None:
        self._widget.toggle_chat_open()
        self._sync_tray_actions()

    def _on_chat_closed(self) -> None:
        self._sync_tray_actions()

    def _on_widget_chat_open_changed(self, opened: bool) -> None:
        if opened:
            self._status_pill.anchor_above(self._widget)
        else:
            self._status_pill.set_text("")
            self._status_pill.hide()
        self._sync_tray_actions()

    def _on_widget_moved(self) -> None:
        if self._status_pill.isVisible():
            self._status_pill.anchor_above(self._widget)

    def _on_chat_status_changed(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._status_pill.set_text(text)
            self._status_pill.anchor_above(self._widget)
            self._status_pill.show()
            self._status_pill.raise_()
        else:
            self._status_pill.set_text("")
            self._status_pill.hide()

    def _handle_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self._toggle_chat_window()

    def _toggle_dashboard(self) -> None:
        self._dashboard.toggle_visibility()

    def _status_snapshot(self) -> dict[str, object]:
        return {
            "backend_ready": self._backend_ready,
            "backend_initializing": self._backend_initializing,
            "backend_error": self._backend_error,
            "signed_in": oauth_codex.is_signed_in(),
            "agent_ready": getattr(self, "_agent", None) is not None,
            "agent_error": getattr(self, "_agent_error", None),
            "mode": self._selected_mode,
            "hotkey": self._hotkey_label,
            "log_path": STARTUP_LOG,
        }

    def _handle_auth_changed(self, signed_in: bool) -> None:
        if signed_in:
            if self._ensure_agent(show_errors=True):
                self.agent_message_received.emit("Signed in. Helper is ready.")
            return

        self._agent = None
        self._agent_error = None
        self._guide_session = None
        self.agent_message_received.emit(
            "Signed out. Chat and computer-use are disabled until you sign in again."
        )

    def _ensure_agent(self, *, show_errors: bool = True) -> bool:
        if self._agent is not None:
            return True
        if not oauth_codex.is_signed_in():
            self._agent_error = "Not signed in. Open Dashboard > Account and sign in with ChatGPT."
            return False
        if self._controller is None or self._dispatcher is None:
            self._agent_error = "Backend is not ready yet."
            return False
        try:
            from agent import HelplerAgent

            self._agent = HelplerAgent(
                capture_provider=self._controller.screenshot,
                dispatcher=self._dispatcher,
            )
            self._agent_error = None
            return True
        except Exception as exc:
            self._agent = None
            self._agent_error = str(exc)
            log.exception("Agent init failed")
            if show_errors:
                self.agent_message_received.emit(f"Could not start Helper agent: {exc}")
            return False

    def _handle_quit_requested(self) -> None:
        self.shutdown()
        self._app.quit()

    def _handle_main_window_close_requested(self) -> None:
        self._handle_quit_requested()

    def _handle_restart_requested(self) -> None:
        log.info("Restart requested. Launching a fresh instance...")
        try:
            subprocess.Popen(
                [sys.executable, *sys.argv],
                close_fds=True,
            )
        except Exception:
            log.exception("Restart failed")
            return
        self._handle_quit_requested()

    def _handle_mode_toggle_requested(self) -> None:
        order = FloatingCircle.MODE_ORDER
        current = self._widget.mode
        idx = order.index(current) if current in order else 0
        target = order[(idx + 1) % len(order)]
        self._request_mode(target)

    def _handle_mode_select_requested(self, mode: str) -> None:
        self._request_mode(mode)

    def _request_mode(self, mode: str) -> None:
        if mode == self._widget.mode:
            return
        if mode == FloatingCircle.ACTIVE and not self._confirm_active_mode():
            return
        self._set_mode(mode)

    def _confirm_active_mode(self) -> bool:
        box = QMessageBox(self._widget)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Enable Active Mode")
        box.setText("Active Mode allows Helper to control the mouse and keyboard.")
        box.setInformativeText(
            "This only applies to the current session. "
            "Destructive actions and terminal Enter still require separate confirmation."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == int(QMessageBox.StandardButton.Yes)

    def _set_mode(self, mode: str) -> None:
        self._selected_mode = mode
        if self._controller is not None:
            self._controller.set_mode(mode)
        self._widget.set_mode(mode)
        self._chat_window.set_mode(mode)
        self.overlay_clear_requested.emit()
        self._ghost_cursor.clear()
        if self._help_session is not None:
            self._help_session.cancel()
        self._guide_session = None
        log.info("Mode set to %s.", mode.upper())

    def _backend_status_message(self) -> str:
        if self._backend_error:
            return f"Helper startup failed: {self._backend_error}"
        if self._backend_initializing:
            return "Helper is still starting up. Try again in a moment."
        if self._agent_error:
            return self._agent_error
        return "Helper backend is not ready yet."

    def _handle_user_message(self, message: str) -> None:
        self._chat_window.add_message(message, True)
        text = (message or "").strip()
        if text:
            self._append_to_active_conversation("user", text)

    def _handle_agent_message(self, message: str) -> None:
        self._chat_window.add_message(message, False)
        text = (message or "").strip()
        if text:
            if self._active_conversation is not None:
                self._append_to_active_conversation("assistant", text)
            self.chat_status_changed.emit(text)

    def _append_to_active_conversation(self, role: str, text: str) -> None:
        if self._active_conversation is None:
            self._active_conversation = StoredConversation.new()
        self._active_conversation.add_message(role, text)
        self._autosave_active_conversation()

    def _autosave_active_conversation(self) -> None:
        conversation = self._active_conversation
        if conversation is None or not conversation.has_user_message():
            return
        conversation.title = conversation.derive_title()
        try:
            self._conversation_store.save(conversation)
        except Exception:
            log.exception("Autosaving conversation failed")

    def _conversations_snapshot(self) -> dict[str, object]:
        try:
            stored = self._conversation_store.load_all()
        except Exception:
            log.exception("Loading conversations failed")
            stored = []

        # Active conversations are autosaved after each message so they survive
        # a crash.  Keep the in-memory copy as the single live row in the
        # dashboard, rather than rendering that same id once from disk and
        # once as LIVE while the session is still open.
        active = self._active_conversation
        if active is not None:
            stored = [conversation for conversation in stored if conversation.id != active.id]

        live = None
        if active is not None and active.has_user_message():
            live = StoredConversation(
                id=active.id,
                started_at=active.started_at,
                ended_at=active.ended_at,
                title=active.derive_title(),
                messages=list(active.messages),
            )
        return {"stored": stored, "live": live}

    def _on_press(self, key: object) -> bool | None:
        if not self._backend_ready or self._audio is None or self._controller is None:
            return None

        name = key_name(key)
        if name is None:
            return None

        if name == "esc":
            with self._lock:
                recording_active = self._recording_active
                if recording_active:
                    self._recording_active = False
                    self._recording_started_at = 0.0
            if recording_active:
                self._audio.cancel_recording()
                self.caption_changed.emit("")
                self.widget_state_requested.emit(FloatingCircle.IDLE)
                log.info("Recording canceled.")
            elif self._chat_window.isVisible():
                self._toggle_chat_window()
            else:
                self._audio.stop_playback()
                self._controller.abort_controller.request_abort("Escape pressed.")
                log.info("Abort requested.")
            return None

        should_start = False
        with self._lock:
            self._pressed.add(name)
            if (
                not self._busy
                and not self._recording_active
                and self._hotkey.issubset(self._pressed)
            ):
                self._recording_active = True
                self._recording_started_at = time.perf_counter()
                should_start = True

        if should_start:
            self._controller.reset_abort()
            self._audio.start_recording()
            self._start_caption_worker()
            self.widget_state_requested.emit(FloatingCircle.LISTENING)
            log.info("Listening...")
        return None

    def _on_release(self, key: object) -> None:
        if not self._backend_ready or self._audio is None:
            return

        name = key_name(key)
        if name is None:
            return

        should_finish = False
        should_toggle_chat = False
        with self._lock:
            self._pressed.discard(name)
            if self._recording_active and not self._hotkey.issubset(self._pressed):
                self._recording_active = False
                elapsed = time.perf_counter() - self._recording_started_at
                self._recording_started_at = 0.0
                if elapsed < CHAT_TAP_THRESHOLD_SEC:
                    should_toggle_chat = True
                else:
                    self._busy = True
                    should_finish = True

        if should_toggle_chat:
            self._audio.cancel_recording()
            self.caption_changed.emit("")
            self.widget_state_requested.emit(FloatingCircle.IDLE)
            log.info("Hotkey tapped. Toggling chat window.")
            self.chat_toggle_requested.emit()
            return

        if should_finish:
            self.widget_state_requested.emit(FloatingCircle.THINKING)
            Thread(target=self._finish_recording, daemon=True).start()

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        if not pressed:
            return
        help_session = self._help_session
        if help_session is None:
            return
        try:
            help_session.notify_user_click(int(x), int(y))
        except Exception:
            log.exception("notify_user_click failed")

    def _start_voice_recording(self) -> None:
        if not self._backend_ready or self._audio is None or self._controller is None:
            return
        should_start = False
        with self._lock:
            if not self._busy and not self._recording_active:
                self._recording_active = True
                self._recording_started_at = time.perf_counter()
                should_start = True
        if should_start:
            self._controller.reset_abort()
            self._audio.start_recording()
            self._start_caption_worker()
            self.widget_state_requested.emit(FloatingCircle.LISTENING)
            log.info("Listening (orb hold)...")

    def _stop_voice_recording(self) -> None:
        if not self._backend_ready or self._audio is None:
            return
        should_finish = False
        with self._lock:
            if self._recording_active:
                self._recording_active = False
                self._recording_started_at = 0.0
                self._busy = True
                should_finish = True
        if should_finish:
            self.widget_state_requested.emit(FloatingCircle.THINKING)
            Thread(target=self._finish_recording, daemon=True).start()

    def _finish_recording(self) -> None:
        if self._audio is None or self._controller is None:
            self._reset_busy_state(clear_caption=True)
            return

        try:
            recording = self._audio.stop_recording()
            log.info("Captured %.2fs. Transcribing...", recording.duration_sec)
            transcript = self._audio.transcribe(recording).strip()
            log.info("[%s] Transcript: %s", self._controller.mode.upper(), transcript)
            if transcript:
                if not self._ensure_agent(show_errors=True):
                    self.agent_message_received.emit(self._backend_status_message())
                    return
                self.user_message_received.emit(transcript)
                if self._selected_mode == FloatingCircle.HELP:
                    self._start_help_walkthrough(transcript)
                else:
                    self.chat_status_changed.emit("Thinking about your request...")
                    self._run_guide_until_done(transcript, speak=True)
        except Exception as exc:
            self._handle_task_exception(exc)
        finally:
            self._reset_busy_state(clear_caption=True)

    def _handle_chat_submitted(self, text: str) -> None:
        message = (text or "").strip()
        if not message:
            return
        if not self._backend_ready or self._controller is None:
            status = self._backend_status_message()
            self._chat_window.add_message(status, False)
            log.info(status)
            return
        if not self._ensure_agent(show_errors=False):
            status = self._backend_status_message()
            self._chat_window.add_message(status, False)
            log.info(status)
            return

        if self._selected_mode == FloatingCircle.HELP:
            self.user_message_received.emit(message)
            self._controller.reset_abort()
            self._start_help_walkthrough(message)
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        self.user_message_received.emit(message)
        self._controller.reset_abort()
        self.widget_state_requested.emit(FloatingCircle.THINKING)
        self.chat_status_changed.emit("Thinking about your request...")
        Thread(target=self._run_chat_message, args=(message,), daemon=True).start()

    def _run_chat_message(self, message: str) -> None:
        if self._controller is None or self._agent is None:
            self._reset_busy_state()
            return

        try:
            log.info("[%s] Chat: %s", self._controller.mode.upper(), message)
            self._run_guide_until_done(message, speak=SPEAK_TYPED_CHAT)
        except Exception as exc:
            self._handle_task_exception(exc)
        finally:
            self._reset_busy_state()

    def _handle_task_exception(self, exc: Exception) -> None:
        if self._abort_requested_type is not None and isinstance(
            exc, self._abort_requested_type
        ):
            log.info("Aborted: %s", exc)
            return

        self.agent_message_received.emit(f"Error: {exc}")
        log.exception("Helper error")

    def _reset_busy_state(self, *, clear_caption: bool = False) -> None:
        with self._lock:
            self._busy = False
        if clear_caption:
            self.caption_changed.emit("")
        self.widget_state_requested.emit(FloatingCircle.IDLE)

    def _start_caption_worker(self) -> None:
        if self._caption_thread is not None and self._caption_thread.is_alive():
            return
        self._caption_thread = Thread(target=self._caption_worker, daemon=True)
        self._caption_thread.start()

    def _caption_worker(self) -> None:
        if self._audio is None:
            return

        interval = 2.0
        min_growth_sec = 0.4
        last_text = ""
        last_duration_sec = 0.0
        while True:
            time.sleep(interval)
            with self._lock:
                still_recording = self._recording_active
            if not still_recording:
                return

            snapshot = self._audio.snapshot_recording()
            if snapshot is None:
                continue

            if snapshot.duration_sec - last_duration_sec < min_growth_sec:
                continue

            try:
                text = self._audio.transcribe(snapshot)
            except Exception:
                continue

            with self._lock:
                if not self._recording_active:
                    return

            last_duration_sec = snapshot.duration_sec
            text = text.strip()
            if text and text != last_text:
                last_text = text
                self.caption_changed.emit(text)

    def _ensure_help_session(self) -> bool:
        if self._help_session is not None:
            return True
        if self._agent is None or self._controller is None:
            return False
        from help_session import HelpSession

        self._help_session = HelpSession(
            self._agent,
            self._controller,
            parent=self,
            overlay_clear_barrier=self._clear_help_overlays_sync,
        )
        self._help_session.ghost_clear.connect(self._ghost_cursor.clear)
        self._help_session.highlight_show.connect(self._on_help_highlight_show)
        self._help_session.highlight_clear.connect(self._overlay.clear)
        self._help_session.target_diagnostic.connect(self._help_target_diagnostics.write)
        self._help_session.chat_message.connect(self._handle_agent_message)
        self._help_session.chat_status.connect(self._on_chat_status_changed)
        self._help_session.finished.connect(self._on_help_session_finished)
        self._help_session.failed.connect(self._on_help_session_failed)
        self._help_session.step_skipped.connect(self._on_help_step_skipped)
        return True

    def _clear_help_overlays_sync(self) -> None:
        ack = Event()
        self.help_overlay_clear_sync_requested.emit(ack)
        if not ack.wait(timeout=0.5):
            log.warning("Timed out waiting for Help overlay clear before capture")

    def _on_help_overlay_clear_sync_requested(self, ack: Event) -> None:
        try:
            self._ghost_cursor.clear()
            self._overlay.clear()
            self._app.processEvents()
        finally:
            ack.set()

    def _on_help_step_skipped(self, message: str) -> None:
        self.chat_status_changed.emit(message)

    def _start_help_walkthrough(self, message: str) -> None:
        if not self._ensure_help_session() or self._help_session is None:
            self.agent_message_received.emit(self._backend_status_message())
            return
        self._ghost_cursor.clear()
        self._overlay.clear()
        log.info("[HELP] %s", message)
        self.widget_state_requested.emit(FloatingCircle.THINKING)
        self.chat_status_changed.emit("Planning your walkthrough...")
        self._help_session.start(message)

    def _on_help_highlight_show(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        label: str,
    ) -> None:
        anchor_x = x + width // 2
        anchor_y = y + height // 2
        self._ghost_cursor.animate_to(anchor_x, anchor_y, label)
        self._overlay.show_highlight(
            x,
            y,
            width,
            height,
            label,
            duration_ms=0,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )

    def _on_help_session_finished(self, _summary: str) -> None:
        self.widget_state_requested.emit(FloatingCircle.IDLE)
        self.chat_status_changed.emit("")

    def _on_help_session_failed(self, message: str) -> None:
        self.agent_message_received.emit(message)
        self.widget_state_requested.emit(FloatingCircle.IDLE)
        self.chat_status_changed.emit("")
        self._ghost_cursor.clear()
        self._overlay.clear()

    def _run_guide_until_done(self, transcript: str, *, speak: bool) -> None:
        if self._agent is None or self._controller is None:
            return

        if self._guide_session is None:
            self._guide_session, turn = self._agent.start_guide(transcript)
        else:
            turn = self._agent.step(self._guide_session, user_text=transcript)

        self._report_turn(turn)
        self._broadcast_turn_status(turn)
        if speak:
            self._speak(self._message_for_turn(turn))

        if self._controller.mode == FloatingCircle.ACTIVE:
            def _surface(t):
                self.widget_state_requested.emit(FloatingCircle.THINKING)
                self._report_turn(t)
                self._broadcast_turn_status(t)
                if speak:
                    self._speak(self._message_for_turn(t))

            turn = run_active_continuation_loop(
                self._agent,
                self._guide_session,
                self._controller.abort_controller,
                turn,
                LOOP_SETTLE_SEC,
                on_turn=_surface,
            )

        if not turn.actions:
            log.info("Turn complete.")

    def _broadcast_turn_status(self, turn: GuideTurn) -> None:
        if turn.actions:
            summary = (turn.actions[0].summary or "").strip()
            if summary:
                self.chat_status_changed.emit(summary)

    def _speak(self, message: str) -> None:
        text = (message or "").strip()
        if (
            not text
            or self._controller is None
            or self._audio is None
            or self._controller.abort_controller.is_aborted()
        ):
            return
        self.widget_state_requested.emit(FloatingCircle.SPEAKING)
        try:
            self._audio.speak_text(text)
        except Exception as exc:
            log.exception("TTS failed: %s", exc)

    def _message_for_turn(self, turn: GuideTurn) -> str:
        message = (turn.message or "").strip()
        if message:
            return message
        if turn.actions:
            return " ".join(action.summary for action in turn.actions[:2]) or "Working on it..."
        return "(no reply)"

    def _report_turn(self, turn: GuideTurn) -> None:
        message = self._message_for_turn(turn)
        log.info("Helper: %s", message)
        self.agent_message_received.emit(message)
        if not turn.actions:
            log.info("Guide turn finished with no further actions.")

    def _handle_overlay_action(self, action: str, payload: dict[str, object]) -> None:
        if self._controller is None:
            return

        screen_x = payload.get("screen_x")
        screen_y = payload.get("screen_y")
        if not isinstance(screen_x, int) or not isinstance(screen_y, int):
            return

        label = str(payload.get("description") or action).strip() or action
        duration = 1500
        self.overlay_highlight_requested.emit(
            screen_x - 24,
            screen_y - 24,
            48,
            48,
            label,
            duration,
            screen_x,
            screen_y,
        )


def _startup_surface(argv: list[str]) -> str:
    args = {arg.strip().lower() for arg in argv[1:]}
    if "--background" in args:
        return "background"
    if "--chat" in args or "--app" in args:
        return "chat"
    return "dashboard"


def _show_fatal_dialog(log_path: Path, exc_text: str) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    box = QMessageBox()
    box.setWindowIcon(helper_app_icon(app))
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Helper failed to start")
    stripped = exc_text.strip()
    first_line = stripped.splitlines()[-1] if stripped else "Unknown error"
    box.setText(first_line)
    box.setDetailedText(f"{exc_text}\n\nLog: {log_path}")
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def main() -> int:
    log_path = setup_logging()
    startup_log = logging.getLogger("helper.main")
    startup_log.info("Helper starting; log=%s", log_path)
    try:
        configure_windows_app_identity()
        app = QApplication(sys.argv)
        app.setApplicationName("Helper")
        app.setApplicationDisplayName("Helper")
        app.setDesktopFileName("Helper")
        app.setQuitOnLastWindowClosed(False)
        helpler = HelplerDesktopApp(app, HOTKEY, startup_surface=_startup_surface(sys.argv))
        helpler.start()
        return app.exec()
    except Exception:
        tb = traceback.format_exc()
        startup_log.error("Fatal startup error\n%s", tb)
        _show_fatal_dialog(log_path, tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
