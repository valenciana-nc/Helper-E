from __future__ import annotations

import argparse
import ctypes
import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Callable, Literal, Sequence

import pyautogui
import pyperclip
from pynput import keyboard

from config import DESTRUCTIVE_KEYWORDS
from screen import Capture, capture_virtual_desktop, set_dpi_aware

Mode = Literal["help", "active"]
OverlayCallback = Callable[[str, dict[str, Any]], None]
ConfirmationCallback = Callable[["ConfirmationRequest"], bool]

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

user32 = ctypes.windll.user32

TERMINAL_CLASS_NAMES = {
    "consolewindowclass",
    "cascadia_hosting_window_class",
    "virtualconsoleclass",
}
TERMINAL_TITLE_KEYWORDS = (
    "powershell",
    "pwsh",
    "cmd.exe",
    "command prompt",
    "windows terminal",
    "terminal",
    "wsl",
    "ubuntu",
    "bash",
    "zsh",
    "python",
)


class AbortRequested(RuntimeError):
    """Raised when Esc or the PyAutoGUI fail-safe aborts the current action."""


def sleep_with_abort(abort_controller: "AbortController", duration: float) -> None:
    deadline = time.perf_counter() + max(duration, 0.0)
    while True:
        abort_controller.checkpoint()
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.02))


class AbortController:
    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._reason = "Action aborted."

    @property
    def reason(self) -> str:
        return self._reason

    def clear(self) -> None:
        with self._lock:
            self._reason = "Action aborted."
            self._event.clear()

    def request_abort(self, reason: str = "Escape pressed.") -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
            self._event.set()

    def is_aborted(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.is_aborted():
            raise AbortRequested(self.reason)


class EscapeAbortListener:
    def __init__(
        self,
        abort_controller: AbortController,
        trigger_key: keyboard.Key = keyboard.Key.esc,
    ) -> None:
        self._abort_controller = abort_controller
        self._listener: keyboard.Listener | None = None
        self._trigger_key = trigger_key

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None

    def _on_press(self, key: keyboard.KeyCode | keyboard.Key | None) -> None:
        if key == self._trigger_key:
            self._abort_controller.request_abort("Escape pressed.")


@dataclass(frozen=True)
class ActionResult:
    action: str
    executed: bool
    mode: Mode
    screen_x: int | None = None
    screen_y: int | None = None
    confirmation_required: bool = False
    confirmation_granted: bool | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class ForegroundWindow:
    hwnd: int
    title: str
    class_name: str


@dataclass(frozen=True)
class ConfirmationRequest:
    action: str
    reason: str
    summary: str
    payload: dict[str, Any]
    foreground_window: ForegroundWindow | None = None


class ComputerController:
    def __init__(
        self,
        mode: Mode = "help",
        *,
        overlay_callback: OverlayCallback | None = None,
        confirmation_callback: ConfirmationCallback | None = None,
        abort_controller: AbortController | None = None,
        pre_action_delay_sec: float = 0.2,
        capture_provider: Callable[[], Capture] = capture_virtual_desktop,
    ) -> None:
        if mode not in ("help", "active"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.overlay_callback = overlay_callback
        self.confirmation_callback = confirmation_callback
        self.abort_controller = abort_controller or AbortController()
        self.pre_action_delay_sec = pre_action_delay_sec
        self.capture_provider = capture_provider

    def set_mode(self, mode: Mode) -> None:
        if mode not in ("help", "active"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def reset_abort(self) -> None:
        self.abort_controller.clear()

    def start_escape_listener(self) -> EscapeAbortListener:
        listener = EscapeAbortListener(self.abort_controller)
        listener.start()
        return listener

    def screenshot(self) -> Capture:
        return self.capture_provider()

    def move_to(
        self,
        x: int,
        y: int,
        *,
        capture: Capture | None = None,
        duration: float = 0.0,
        description: str = "",
        requires_confirmation: bool = False,
        confirmation_reason: str = "",
    ) -> ActionResult:
        screen_x, screen_y = self._translate_point(x, y, capture)
        payload = {
            "x": x,
            "y": y,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "duration": duration,
            "description": description,
            "requires_confirmation": requires_confirmation,
            "confirmation_reason": confirmation_reason,
        }
        return self._dispatch(
            "move_to",
            payload,
            lambda: pyautogui.moveTo(screen_x, screen_y, duration=duration),
        )

    def click(
        self,
        x: int,
        y: int,
        *,
        capture: Capture | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
        description: str = "",
        requires_confirmation: bool = False,
        confirmation_reason: str = "",
    ) -> ActionResult:
        screen_x, screen_y = self._translate_point(x, y, capture)
        payload = {
            "x": x,
            "y": y,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "button": button,
            "clicks": clicks,
            "description": description,
            "requires_confirmation": requires_confirmation,
            "confirmation_reason": confirmation_reason,
        }
        return self._dispatch(
            "click",
            payload,
            lambda: pyautogui.click(
                x=screen_x,
                y=screen_y,
                clicks=clicks,
                interval=interval,
                button=button,
            ),
        )

    def drag_to(
        self,
        x: int,
        y: int,
        *,
        capture: Capture | None = None,
        button: str = "left",
        duration: float = 0.2,
        description: str = "",
        requires_confirmation: bool = False,
        confirmation_reason: str = "",
    ) -> ActionResult:
        screen_x, screen_y = self._translate_point(x, y, capture)
        payload = {
            "x": x,
            "y": y,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "button": button,
            "duration": duration,
            "description": description,
            "requires_confirmation": requires_confirmation,
            "confirmation_reason": confirmation_reason,
        }
        return self._dispatch(
            "drag_to",
            payload,
            lambda: pyautogui.dragTo(screen_x, screen_y, duration=duration, button=button),
        )

    def type_text(
        self,
        text: str,
        *,
        interval: float = 0.0,
        description: str = "",
        requires_confirmation: bool = False,
        confirmation_reason: str = "",
    ) -> ActionResult:
        payload = {
            "text": text,
            "interval": interval,
            "description": description,
            "requires_confirmation": requires_confirmation,
            "confirmation_reason": confirmation_reason,
        }
        return self._dispatch(
            "type_text",
            payload,
            lambda: self._type_text_executor(text, interval),
        )

    def key(
        self,
        combo: str | Sequence[str],
        *,
        description: str = "",
        requires_confirmation: bool = False,
        confirmation_reason: str = "",
        force_execute: bool = False,
    ) -> ActionResult:
        keys = self._normalize_keys(combo)
        if not keys:
            raise ValueError("Expected at least one key.")

        payload = {
            "keys": keys,
            "description": description,
            "requires_confirmation": requires_confirmation,
            "confirmation_reason": confirmation_reason,
            "force_execute": force_execute,
        }
        return self._dispatch(
            "key",
            payload,
            lambda: pyautogui.press(keys[0]) if len(keys) == 1 else pyautogui.hotkey(*keys),
        )

    def scroll(
        self,
        dy: int,
        *,
        x: int | None = None,
        y: int | None = None,
        capture: Capture | None = None,
        description: str = "",
        requires_confirmation: bool = False,
        confirmation_reason: str = "",
    ) -> ActionResult:
        screen_x: int | None = None
        screen_y: int | None = None
        if x is not None and y is not None:
            screen_x, screen_y = self._translate_point(x, y, capture)

        payload = {
            "dy": dy,
            "x": x,
            "y": y,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "description": description,
            "requires_confirmation": requires_confirmation,
            "confirmation_reason": confirmation_reason,
        }
        return self._dispatch(
            "scroll",
            payload,
            lambda: pyautogui.scroll(dy, x=screen_x, y=screen_y),
        )

    def _dispatch(
        self,
        action: str,
        payload: dict[str, Any],
        executor: Callable[[], None],
    ) -> ActionResult:
        self.abort_controller.checkpoint()

        confirmation_request = self._build_confirmation_request(action, payload)
        if confirmation_request is not None:
            approved = False
            if self.confirmation_callback is not None:
                approved = self.confirmation_callback(confirmation_request)

            if not approved:
                return ActionResult(
                    action=action,
                    executed=False,
                    mode=self.mode,
                    screen_x=payload.get("screen_x"),
                    screen_y=payload.get("screen_y"),
                    confirmation_required=True,
                    confirmation_granted=False,
                    blocked_reason=confirmation_request.reason,
                )

        self._sleep_with_abort(self.pre_action_delay_sec)
        self.abort_controller.checkpoint()

        try:
            executor()
        except pyautogui.FailSafeException as exc:
            self.abort_controller.request_abort("PyAutoGUI fail-safe triggered.")
            raise AbortRequested(self.abort_controller.reason) from exc

        return ActionResult(
            action=action,
            executed=True,
            mode=self.mode,
            screen_x=payload.get("screen_x"),
            screen_y=payload.get("screen_y"),
            confirmation_required=confirmation_request is not None,
            confirmation_granted=True if confirmation_request is not None else None,
        )

    def _build_confirmation_request(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> ConfirmationRequest | None:
        if payload.get("requires_confirmation"):
            summary = self._summarize_action(action, payload)
            reason = str(payload.get("confirmation_reason") or "Model requested confirmation.")
            return ConfirmationRequest(
                action=action,
                reason=reason,
                summary=summary,
                payload=dict(payload),
                foreground_window=self._foreground_window(),
            )

        dangerous_text = self._dangerous_text(action, payload)
        if dangerous_text is not None:
            summary = self._summarize_action(action, payload)
            reason = (
                "Potentially destructive content detected in the requested text."
                if action == "type_text"
                else f"Potentially destructive content detected: {dangerous_text!r}."
            )
            return ConfirmationRequest(
                action=action,
                reason=reason,
                summary=summary,
                payload=dict(payload),
                foreground_window=self._foreground_window(),
            )

        if action == "key" and self._contains_enter_key(payload.get("keys")):
            window = self._foreground_window()
            if self._is_terminal_window(window):
                summary = self._summarize_action(action, payload)
                return ConfirmationRequest(
                    action=action,
                    reason="Enter key would be sent to a terminal window.",
                    summary=summary,
                    payload=dict(payload),
                    foreground_window=window,
                )

        return None

    def _dangerous_text(self, action: str, payload: dict[str, Any]) -> str | None:
        candidates: list[str] = []
        description = str(payload.get("description") or "").strip()
        if description:
            candidates.append(description)

        if action == "type_text":
            text = str(payload.get("text") or "").strip()
            if text:
                candidates.append(text)

        for candidate in candidates:
            normalized = f" {candidate.lower()} "
            for keyword in DESTRUCTIVE_KEYWORDS:
                if keyword.lower() in normalized:
                    return candidate
        return None

    @staticmethod
    def _contains_enter_key(keys: Any) -> bool:
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            return False
        normalized = {str(key).strip().lower() for key in keys if str(key).strip()}
        return any(key in normalized for key in {"enter", "return"})

    @staticmethod
    def _is_terminal_window(window: ForegroundWindow | None) -> bool:
        if window is None:
            return False

        class_name = window.class_name.strip().lower()
        title = window.title.strip().lower()
        if class_name in TERMINAL_CLASS_NAMES:
            return True
        return any(keyword in title for keyword in TERMINAL_TITLE_KEYWORDS)

    @staticmethod
    def _summarize_action(action: str, payload: dict[str, Any]) -> str:
        if action in {"move_to", "click", "drag_to"}:
            x = payload.get("screen_x")
            y = payload.get("screen_y")
            description = str(payload.get("description") or "").strip()
            suffix = f" ({description})" if description else ""
            return f"{action} at ({x}, {y}){suffix}"

        if action == "type_text":
            text = str(payload.get("text") or "")
            return f"type_text: {len(text)} characters"

        if action == "key":
            keys = payload.get("keys") or []
            return f"key: {'+'.join(str(key) for key in keys)}"

        if action == "scroll":
            return f"scroll: dy={payload.get('dy')}"

        return action

    @staticmethod
    def _foreground_window() -> ForegroundWindow | None:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        title_buffer = ctypes.create_unicode_buffer(512)
        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
        user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))
        return ForegroundWindow(
            hwnd=int(hwnd),
            title=title_buffer.value,
            class_name=class_buffer.value,
        )

    @staticmethod
    def _type_text_executor(text: str, interval: float) -> None:
        if text.isascii() and len(text) <= 80:
            pyautogui.write(text, interval=interval)
            return
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

    def _sleep_with_abort(self, duration: float) -> None:
        sleep_with_abort(self.abort_controller, duration)

    @staticmethod
    def _translate_point(x: int, y: int, capture: Capture | None) -> tuple[int, int]:
        if capture is None:
            return int(x), int(y)
        return capture.to_screen_coords(int(x), int(y))

    @staticmethod
    def _normalize_keys(combo: str | Sequence[str]) -> list[str]:
        if isinstance(combo, str):
            return [part.strip().lower() for part in combo.split("+") if part.strip()]
        return [str(part).strip().lower() for part in combo if str(part).strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper computer-control smoke test.")
    parser.add_argument("--x", type=int, help="X coordinate in capture space.")
    parser.add_argument("--y", type=int, help="Y coordinate in capture space.")
    parser.add_argument(
        "--active",
        action="store_true",
        help="Execute the click instead of emitting a guide-mode preview.",
    )
    parser.add_argument(
        "--button",
        default="left",
        choices=("left", "middle", "right"),
        help="Mouse button to use for click tests.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    set_dpi_aware()
    controller = ComputerController(
        mode="active" if args.active else "help",
        overlay_callback=lambda action, payload: print(f"{action}: {payload}"),
    )
    listener = controller.start_escape_listener()

    try:
        capture = controller.screenshot()
        print(
            "Captured "
            f"{capture.width}x{capture.height} at scale {capture.scale:.3f} "
            f"from origin ({capture.monitor_left}, {capture.monitor_top})"
        )

        if args.x is None or args.y is None:
            print("Pass --x and --y to test translated clicks.")
        else:
            result = controller.click(args.x, args.y, capture=capture, button=args.button)
            print(result)
    finally:
        listener.stop()
