"""AnthropicProvider — talks to the Anthropic Messages API.

Used when HELPER_PROVIDER=anthropic. Reuses ChatResult/ToolCall/ProviderError
types and retry helpers from openai_client so the rest of the codebase doesn't
need to know which provider it's talking to.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import requests

from openai_client import (
    BadProviderResponse,
    CONNECT_TIMEOUT_SEC,
    READ_TIMEOUT_SEC,
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    AuthExpired,
    ChatResult,
    COMPUTER_USE_TOOLS,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    ROUTE_CLASSIFIER_PROMPT,
    ToolCall,
    _redact,
)

log = logging.getLogger("helper.anthropic")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Sampling parameters were removed from these model families; sending
# `temperature` returns HTTP 400. Matched by prefix so dated and future
# point-release IDs are covered. Extend when Anthropic ships new families.
SAMPLING_UNSUPPORTED_MODEL_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable",
    "claude-mythos",
)


def model_accepts_temperature(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return not normalized.startswith(SAMPLING_UNSUPPORTED_MODEL_PREFIXES)


class AnthropicProvider:
    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public API matching ChatCompletionsProvider / CodexProvider surface.

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        body = self._body(
            messages,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _parse_messages(self._post(body))

    def computer_use_step(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        body = self._body(
            messages,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        body["tools"] = _anthropic_tools(COMPUTER_USE_TOOLS)
        body["tool_choice"] = {"type": "auto"}
        return _parse_messages(self._post(body))

    def classify_route(self, text: str, *, model: str) -> str:
        body = {
            "model": model,
            "max_tokens": 8,
            "system": ROUTE_CLASSIFIER_PROMPT,
            "messages": [
                {"role": "user", "content": (text or "").strip()[:500]},
            ],
        }
        try:
            payload = self._post(body)
        except ProviderError as exc:
            log.warning("Anthropic route classifier failed: %s", exc)
            return ""
        try:
            result = _parse_messages(payload)
        except BadProviderResponse as exc:
            log.warning("Anthropic route classifier returned unreadable response: %s", exc)
            return ""
        words = (result.text or "").strip().lower().split()
        if not words:
            return ""
        return words[0].strip(".,'\"!?")

    # ------------------------------------------------------------------
    # Internals.

    def _body(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        translated, extracted_system = _to_anthropic_messages(messages)
        system_text = system_prompt or extracted_system or None
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max(1, int(max_tokens or 4096)),
            "messages": translated,
        }
        if system_text:
            body["system"] = system_text
        if temperature is not None and model_accepts_temperature(model):
            body["temperature"] = float(temperature)
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        last_error: ProviderError | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    ANTHROPIC_URL,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=(CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC),
                )
            except requests.Timeout:
                last_error = ProviderUnavailable("Anthropic request timed out.")
            except requests.RequestException as exc:
                last_error = ProviderUnavailable(f"Anthropic request failed: {exc}")
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise BadProviderResponse(
                            "Anthropic returned non-JSON response."
                        ) from exc

                if resp.status_code in {401, 403}:
                    raise AuthExpired(
                        "Anthropic rejected the key (HTTP "
                        f"{resp.status_code}). Check HELPER_ANTHROPIC_API_KEY."
                    )

                detail = _redact((resp.text or "")[:400])
                message = f"Anthropic failed: HTTP {resp.status_code}: {detail}"
                if resp.status_code == 429:
                    last_error = RateLimited(message)
                elif resp.status_code in RETRYABLE_STATUS_CODES:
                    last_error = ProviderUnavailable(message)
                else:
                    raise ProviderError(message)

                log.warning("Anthropic HTTP %s attempt=%s body=%s", resp.status_code, attempt + 1, detail)

            if attempt < MAX_RETRIES:
                time.sleep(0.6 * (2 ** attempt))

        if last_error is not None:
            raise last_error
        raise ProviderUnavailable("Anthropic request failed.")


# ---------------------------------------------------------------------------
# Translation helpers.

def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Translate Helper's internal message shape to Anthropic's.

    Returns (messages, system_text). system_text is the concatenation of any
    role=system messages in the history — callers can override with their own.
    """
    out: list[dict[str, Any]] = []
    system_chunks: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        parts = msg.get("parts") or []

        if role == "system":
            for part in parts:
                if "text" in part and part["text"]:
                    system_chunks.append(part["text"])
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            for part in parts:
                if "text" in part and part["text"]:
                    blocks.append({"type": "text", "text": part["text"]})
                elif "function_call" in part:
                    fc = part["function_call"]
                    blocks.append({
                        "type": "tool_use",
                        "id": fc.get("call_id") or f"call_{fc.get('name', 'tool')}",
                        "name": fc.get("name", ""),
                        "input": fc.get("arguments") or {},
                    })
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue

        # user / tool turns — Anthropic represents tool results inside user blocks.
        blocks = []
        for part in parts:
            if "text" in part and part["text"]:
                blocks.append({"type": "text", "text": part["text"]})
            elif "image_png" in part and part["image_png"]:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(part["image_png"]).decode("ascii"),
                    },
                })
            elif "function_response" in part:
                fr = part["function_response"]
                response_obj = fr.get("response")
                if isinstance(response_obj, str):
                    payload = response_obj
                else:
                    payload = json.dumps(response_obj or {})
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": fr.get("call_id") or f"call_{fr.get('name', 'tool')}",
                    "content": payload,
                })
        if blocks:
            out.append({"role": "user", "content": blocks})

    system_text = "\n\n".join(s for s in system_chunks if s) or None
    return out, system_text


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wrapped: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        wrapped.append({
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
        })
    return wrapped


def _parse_messages(payload: dict[str, Any]) -> ChatResult:
    if not isinstance(payload, dict):
        raise BadProviderResponse("Anthropic returned a non-object response.")
    content = payload.get("content")
    if not isinstance(content, list):
        raise BadProviderResponse("Anthropic response has no content blocks.")

    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text") or ""
            if isinstance(t, str) and t.strip():
                text_chunks.append(t.strip())
        elif btype == "tool_use":
            raw_args = block.get("input") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            tool_calls.append(ToolCall(
                name=str(block.get("name") or ""),
                arguments=raw_args if isinstance(raw_args, dict) else {},
                call_id=block.get("id"),
            ))

    return ChatResult(text=" ".join(text_chunks).strip(), tool_calls=tool_calls)
