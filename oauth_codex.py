from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

import token_store
from token_store import TokenSet

log = logging.getLogger("helper.oauth")

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 1455
REDIRECT_PATH = "/auth/callback"
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}{REDIRECT_PATH}"
SCOPES = "openid profile email offline_access"
REFRESH_LEEWAY_SEC = 300

_refresh_lock = threading.Lock()


class LoginError(RuntimeError):
    pass


class NotSignedIn(RuntimeError):
    pass


@dataclass
class LoginHandle:
    server: HTTPServer
    server_thread: threading.Thread
    completion: threading.Event
    result: dict[str, str]
    state: str
    code_verifier: str

    def wait(self, timeout: float = 180.0) -> TokenSet:
        try:
            if not self.completion.wait(timeout):
                raise LoginError("Timed out waiting for the browser callback.")
            if "error" in self.result:
                raise LoginError(f"OAuth error: {self.result.get('error_description', self.result['error'])}")
            if self.result.get("state") != self.state:
                raise LoginError("OAuth state mismatch — possible CSRF; aborting.")
            code = self.result.get("code")
            if not code:
                raise LoginError("OAuth callback did not return a code.")
            tokens = _exchange_code(code, self.code_verifier)
            token_store.save(tokens)
            return tokens
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass


def start_login() -> LoginHandle:
    code_verifier = _make_code_verifier()
    code_challenge = _make_code_challenge(code_verifier)
    state = secrets.token_urlsafe(24)

    result: dict[str, str] = {}
    completion = threading.Event()

    handler = _build_handler(result, completion)
    try:
        server = HTTPServer((REDIRECT_HOST, REDIRECT_PORT), handler)
    except OSError as exc:
        raise LoginError(
            f"Could not start the local sign-in callback on {REDIRECT_HOST}:{REDIRECT_PORT}. "
            "Close any other Helper sign-in window or app using that port, then try again."
        ) from exc
    thread = threading.Thread(target=server.serve_forever, name="helper-oauth-cb", daemon=True)
    thread.start()

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    auth_url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    log.info("Opening browser for ChatGPT sign-in: %s", AUTHORIZE_URL)
    if not webbrowser.open(auth_url):
        server.shutdown()
        server.server_close()
        raise LoginError("Could not open the browser for ChatGPT sign-in.")

    return LoginHandle(
        server=server,
        server_thread=thread,
        completion=completion,
        result=result,
        state=state,
        code_verifier=code_verifier,
    )


def get_access_token() -> str:
    tokens = token_store.load()
    if tokens is None:
        raise NotSignedIn("Not signed in. Click 'Sign in with ChatGPT' in the dashboard.")

    if tokens.expires_at - time.time() < REFRESH_LEEWAY_SEC:
        tokens = _refresh(tokens)
    return tokens.access_token


def sign_out() -> None:
    token_store.clear()


def is_signed_in() -> bool:
    return token_store.load() is not None


def account_email() -> str | None:
    tokens = token_store.load()
    if tokens is None or not tokens.id_token:
        return None
    payload = _decode_jwt_payload(tokens.id_token)
    if not isinstance(payload, dict):
        return None
    return payload.get("email") or payload.get("preferred_username")


def _exchange_code(code: str, code_verifier: str) -> TokenSet:
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": REDIRECT_URI,
    }
    try:
        resp = requests.post(TOKEN_URL, data=data, timeout=30)
    except requests.RequestException as exc:
        raise LoginError(f"Token exchange could not reach OpenAI auth: {exc}") from exc
    if resp.status_code != 200:
        raise LoginError(f"Token exchange failed: HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise LoginError("Token exchange returned invalid JSON.") from exc
    return _token_set_from_response(payload, prev_refresh=None)


def _refresh(prev: TokenSet) -> TokenSet:
    with _refresh_lock:
        current = token_store.load() or prev
        if current.expires_at - time.time() >= REFRESH_LEEWAY_SEC:
            return current

        data = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": current.refresh_token,
            "scope": SCOPES,
        }
        if not current.refresh_token:
            token_store.clear()
            raise LoginError("Your ChatGPT sign-in expired. Please sign in again.")
        try:
            resp = requests.post(TOKEN_URL, data=data, timeout=30)
        except requests.RequestException as exc:
            raise LoginError(f"Token refresh could not reach OpenAI auth: {exc}") from exc
        if resp.status_code != 200:
            if resp.status_code in {400, 401, 403}:
                token_store.clear()
                raise LoginError("Your ChatGPT sign-in expired. Please sign in again.")
            raise LoginError(f"Token refresh failed: HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise LoginError("Token refresh returned invalid JSON.") from exc
        tokens = _token_set_from_response(payload, prev_refresh=current.refresh_token)
        token_store.save(tokens)
        return tokens


def _token_set_from_response(payload: dict, *, prev_refresh: str | None) -> TokenSet:
    if not isinstance(payload, dict):
        raise LoginError("Token response was not a JSON object.")

    access = payload.get("access_token")
    if not isinstance(access, str) or not access.strip():
        raise LoginError("Token response missing access_token.")

    try:
        expires_in = float(payload.get("expires_in", 3600))
    except (TypeError, ValueError, OverflowError) as exc:
        raise LoginError("Token response contained an invalid expires_in value.") from exc
    if not math.isfinite(expires_in) or expires_in <= 0:
        raise LoginError("Token response contained an invalid expires_in value.")

    refresh = payload.get("refresh_token")
    if not isinstance(refresh, str):
        refresh = prev_refresh or ""
    id_token = payload.get("id_token")
    if not isinstance(id_token, str):
        id_token = ""
    account_id = None
    if id_token:
        claims = _decode_jwt_payload(id_token)
        if isinstance(claims, dict):
            auth_claim = claims.get("https://api.openai.com/auth", {})
            if isinstance(auth_claim, dict):
                account_id = auth_claim.get("chatgpt_account_id") or auth_claim.get("chatgpt_user_id")
    return TokenSet(
        access_token=access,
        refresh_token=refresh,
        id_token=id_token,
        expires_at=time.time() + expires_in,
        account_id=account_id,
    )


def _make_code_verifier() -> str:
    return secrets.token_urlsafe(64)[:128]


def _make_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _decode_jwt_payload(jwt: str):
    try:
        _, payload_b64, _ = jwt.split(".")
    except ValueError:
        return None
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None


def _build_handler(result: dict[str, str], completion: threading.Event):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != REDIRECT_PATH:
                self.send_response(404)
                self.end_headers()
                return

            params = dict(urllib.parse.parse_qsl(parsed.query))
            result.update(params)

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if "error" in params:
                body = f"<h1>Sign-in failed</h1><p>{params.get('error_description', params['error'])}</p>"
            else:
                body = "<h1>Signed in</h1><p>You can close this tab and return to Helper.</p>"
            self.wfile.write(body.encode("utf-8"))
            completion.set()

        def log_message(self, format, *args):  # noqa: A002
            log.debug("oauth-callback: " + format, *args)

    return _Handler
