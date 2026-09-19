"""Safe, shell-free launching for desktop actions.

Model output and user-entered URLs are data, not command lines.  Keeping the
launch implementation in one module prevents one action path from silently
falling back to ``cmd /c`` while another path is hardened.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit


class LaunchError(RuntimeError):
    """Raised when a desktop target cannot be launched safely."""


_SHELL_METACHARACTERS = frozenset("&|<>^`\r\n")
_WEB_SCHEMES = frozenset({"http", "https"})
HELP_LAUNCH_COMMANDS = frozenset(
    {"chrome", "msedge", "notepad", "calc", "explorer", "taskmgr", "wt"}
)


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _launch_uri(target: str) -> None:
    parsed = urlsplit(target)
    if parsed.scheme.lower() not in _WEB_SCHEMES or not parsed.netloc:
        raise LaunchError("Only valid http(s) URLs may be opened as web targets.")
    if not webbrowser.open(target, new=0, autoraise=True):
        raise LaunchError("The default browser declined the URL.")


def safe_target_label(target: str) -> str:
    """Return a log/UI-safe description without query strings or credentials."""
    value = str(target or "").strip()
    if not value:
        return "the requested target"
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "the requested target"
    if parsed.scheme.lower() in _WEB_SCHEMES and parsed.netloc:
        path = parsed.path or "/"
        host = parsed.hostname or ""
        try:
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
        except ValueError:
            return f"{parsed.scheme.lower()}://{host}{path}"
        return f"{parsed.scheme.lower()}://{host}{path}"
    if parsed.scheme.lower() == "ms-settings":
        return "Windows Settings"
    try:
        argv = shlex.split(value, posix=False)
    except ValueError:
        return "the requested app"
    return argv[0] if argv else "the requested app"


def launch_target(target: str, *, allowed_commands: frozenset[str] | None = None) -> None:
    """Launch a URL, Windows shell URI, file, or executable without a shell."""
    value = str(target or "").strip()
    if not value:
        raise LaunchError("No launch target was provided.")
    if any(char in value for char in _SHELL_METACHARACTERS):
        raise LaunchError("Launch target contains shell metacharacters.")

    parsed = urlsplit(value)
    if value.lower() == "about:blank":
        if not webbrowser.open(value, new=0, autoraise=True):
            raise LaunchError("The default browser declined the blank page.")
        return
    if parsed.scheme.lower() in _WEB_SCHEMES:
        _launch_uri(value)
        return

    # Windows shell URIs (for example ms-settings:) and existing files/folders
    # are opened by the OS without invoking a command interpreter.
    if parsed.scheme.lower() == "ms-settings" or Path(value).exists():
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise LaunchError("Windows shell launching is unavailable on this platform.")
        startfile(value)
        return

    try:
        argv = shlex.split(value, posix=False)
    except ValueError as exc:
        raise LaunchError("Launch target has invalid quoting.") from exc
    if not argv:
        raise LaunchError("No launch command was provided.")
    if allowed_commands is not None:
        executable = Path(argv[0]).name.lower()
        if executable not in allowed_commands:
            raise LaunchError("This application is not allowed as an automatic helper launch.")

    subprocess.Popen(
        argv,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_creation_flags(),
    )
