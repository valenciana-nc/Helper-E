from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import keyring

log = logging.getLogger("helper.token_store")

SERVICE = "Helper"
LEGACY_SERVICES = ("Helpler", "Harvis")
USERNAME = "openai-codex"
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"


@dataclass(frozen=True)
class TokenSet:
    access_token: str
    refresh_token: str
    id_token: str
    expires_at: float
    account_id: str | None = None


def save(tokens: TokenSet) -> None:
    keyring.set_password(SERVICE, USERNAME, json.dumps(asdict(tokens)))


def load() -> TokenSet | None:
    current = _load_from_service(SERVICE)
    if current is not None:
        return current

    for legacy_service in LEGACY_SERVICES:
        legacy = _load_from_service(legacy_service)
        if legacy is None:
            continue
        try:
            save(legacy)
            log.info("Migrated ChatGPT token from %s to %s.", legacy_service, SERVICE)
        except Exception as exc:
            log.warning("Could not migrate token from %s: %s", legacy_service, exc)
        return legacy

    codex = _load_from_codex_auth(CODEX_AUTH_PATH)
    if codex is not None:
        log.debug("Using ChatGPT token from %s.", CODEX_AUTH_PATH)
        return codex
    return None


def clear() -> None:
    for service in (SERVICE, *LEGACY_SERVICES):
        try:
            keyring.delete_password(service, USERNAME)
        except keyring.errors.PasswordDeleteError:
            pass
        except Exception as exc:
            log.warning("Could not clear token from %s: %s", service, exc)


def _load_from_service(service: str) -> TokenSet | None:
    try:
        raw = keyring.get_password(service, USERNAME)
    except Exception as exc:
        # Headless/CI machines may not have a usable keyring backend. Treat
        # that as signed-out instead of preventing the desktop app from
        # starting.
        log.warning("Could not read token from %s: %s", service, exc)
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return TokenSet(**data)
    except (ValueError, TypeError) as exc:
        log.warning(
            "Stored token blob for %s is unreadable, treating as signed-out: %s",
            service,
            exc,
        )
        return None


def _load_from_codex_auth(path: Path) -> TokenSet | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read Codex auth file %s: %s", path, exc)
        return None

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return None

    access = str(tokens.get("access_token") or "")
    refresh = str(tokens.get("refresh_token") or "")
    id_token = str(tokens.get("id_token") or "")
    if not access:
        return None

    account_id = tokens.get("account_id")
    if not isinstance(account_id, str) or not account_id:
        account_id = _account_id_from_id_token(id_token)

    return TokenSet(
        access_token=access,
        refresh_token=refresh,
        id_token=id_token,
        expires_at=_jwt_exp(access) or time.time() + 3600,
        account_id=account_id,
    )


def _account_id_from_id_token(id_token: str) -> str | None:
    claims = _decode_jwt_payload(id_token)
    if not isinstance(claims, dict):
        return None
    auth_claim = claims.get("https://api.openai.com/auth", {})
    if not isinstance(auth_claim, dict):
        return None
    value = auth_claim.get("chatgpt_account_id") or auth_claim.get("chatgpt_user_id")
    return value if isinstance(value, str) and value else None


def _jwt_exp(jwt: str) -> float | None:
    claims = _decode_jwt_payload(jwt)
    if not isinstance(claims, dict):
        return None
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def _decode_jwt_payload(jwt: str) -> dict[str, Any] | None:
    import base64

    try:
        _, payload_b64, _ = jwt.split(".")
    except ValueError:
        return None
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None
