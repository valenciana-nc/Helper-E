from __future__ import annotations

from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        values[key.strip()] = _strip_quotes(value.strip())
    return values


def write_env(path: Path, updates: dict[str, str]) -> None:
    if not updates:
        return

    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    out_lines: list[str] = []

    for raw in existing_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(raw)
            continue
        key, sep, _ = raw.partition("=")
        key_clean = key.strip()
        if sep and key_clean in remaining:
            out_lines.append(f"{key_clean}={_format_value(remaining.pop(key_clean))}")
        else:
            out_lines.append(raw)

    if remaining:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        for key, value in remaining.items():
            out_lines.append(f"{key}={_format_value(value)}")

    text = "\n".join(out_lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _strip_quotes(value: str) -> str:
    """Decode the small, deliberately conservative ``.env`` value format.

    ``write_env`` quotes values containing whitespace or comment characters.
    Double-quoted values therefore need the inverse of its escaping rules;
    without this, an API key containing a quote was persisted in a different
    form every time the dashboard saved settings.
    """
    value = value.strip()
    if len(value) < 2 or value[0] not in ("'", '"'):
        return value

    quote = value[0]
    if quote == "'":
        if value.endswith("'"):
            return value[1:-1]
        return value

    if not value.endswith('"'):
        return value

    decoded: list[str] = []
    escaped = False
    for char in value[1:-1]:
        if escaped:
            if char in ('"', "\\"):
                decoded.append(char)
            else:
                # Preserve unknown escape sequences instead of silently
                # changing user data.
                decoded.extend(("\\", char))
            escaped = False
            continue
        if char == "\\":
            escaped = True
        else:
            decoded.append(char)
    if escaped:
        decoded.append("\\")
    return "".join(decoded)


def _format_value(value: str) -> str:
    if value == "":
        return ""
    if any(ch in value for ch in (" ", "\t", "#", "'", '"', "\\")):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value
