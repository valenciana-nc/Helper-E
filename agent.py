from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Protocol

from config import (
    AGENT_MODEL,
    AGENT_TIMEOUT_SEC,
    MAX_AGENT_STEPS,
    MAX_TOKENS,
    OPENAI_API_KEY,
    REASONING_MODEL,
    TEMPERATURE,
    USE_ROUTE_CLASSIFIER,
)
from fast_actions import (
    APP_ALIASES,
    HOTKEY_ALIASES,
    SITE_ALIASES,
    parse_fast_action,
)
from control_inventory import ControlCandidate, format_candidates_for_prompt
from history import HistoryManager
from launch_safety import safe_target_label
from openai_client import ChatResult, OpenAIClient, ToolCall, make_openai_client
from screen import Capture, capture_virtual_desktop

log = logging.getLogger("helper.agent")

CHAT_SYSTEM_PROMPT = """
You are Helper, a conversational Windows 11 desktop assistant.

Behavior:
- Answer as a concise, helpful assistant for normal conversation.
- Do not claim to click, type, open, or otherwise operate the desktop in this mode.
- If the user is asking you to directly control the computer, briefly say you need computer-use mode for that task.
- Keep replies direct and natural.
""".strip()

COMPUTER_USE_SYSTEM_PROMPT = """
You are Helper, a conversational Windows 11 desktop assistant.

The user provides a live desktop screenshot with each turn. You can answer as a normal chat assistant, and you can
also call the provided desktop-action tools when the user wants help inspecting or operating the desktop.

Behavior:
- Default to a concise, helpful conversational reply when no on-screen action is needed.
- If the user asked Helper to do something on the desktop, you MUST call at least one tool. Do not narrate intent
  ("I'll open Gmail for you") without acting — describe nothing, just call the tool.
- When you call tools, suggest the next 1-3 tightly-coupled actions that do NOT depend on the screen updating between
  them (e.g., a click followed by typing into the same field). STOP and let Helper re-screenshot after any action that
  opens a menu, dialog, page, or new window. Never queue actions on a UI state you can't see yet.
- Never assume a suggested action has already been executed unless the tool result or user explicitly confirms it.
- If the task is complete or no UI action is needed, answer with plain text only and no tool calls.
- Keep natural-language explanations direct and brief.
- If an action looks destructive, risky, sends data, or confirms a transaction, include a safety_decision argument
  with decision="require_confirmation" and a short explanation.
- Coordinates are normalized 0-1000 over the provided screenshot.
""".strip()

HELP_SYSTEM_PROMPT = """
You are Helper running in Help mode. The user wants to LEARN how to complete a task on their Windows 11 desktop.
Your job is to produce a guided walkthrough: a sequence of short steps that the user will follow themselves,
while Helper visually points at each target with a ghost cursor + highlight rectangle + caption.

Helper can scroll, send keyboard shortcuts, and launch apps / folders / URLs between steps to set the scene.
Helper MUST NEVER click any UI button — clicking is the user's action to perform. You teach; the user clicks.

You receive the user's current screenshot. Coordinates are normalized 0-1000 over that screenshot.

Respond with JSON only, no prose outside the JSON. Use this exact shape:
{
  "plan_summary": "one short sentence introducing the walkthrough",
  "fallback_text": "",
  "steps": [
    {
      "instruction": "Short friendly direction shown next to the ghost cursor (one sentence).",
      "target": {"x": 0-1000, "y": 0-1000, "width": 0-1000, "height": 0-1000},
      "expected_change": "What the user should see on screen once this step is done.",
      "helper_action": null
    }
  ]
}

helper_action (optional, runs BEFORE the step is shown to the user) must be one of:
  {"name": "launch_app", "command": "ms-settings:", "display_name": "Settings"}
  {"name": "open_url", "url": "https://supabase.com/dashboard"}
  {"name": "key", "keys": "ctrl+t"}
  {"name": "scroll", "direction": "down" | "up"}
Never include a click action; never include type_text_at, click_at, drag_and_drop, click_control.

Rules:
- 3 to 8 steps total. Each step is ONE concrete user action.
- target must be a TIGHT bounding box around the EXACT clickable element (button, menu item, tab,
  icon, link, or input field). It must NOT enclose the whole panel, the whole window, or empty space.
  Typical rect size in normalized units: width 20-120 and height 12-60 for a button or menu item;
  width 40-300 and height 16-40 for an input field. Centered on the element the user must click.
- Re-check that the element described in `instruction` is the one inside `target` before emitting it.
  If you cannot localize the element confidently, prefer fewer, more accurate steps over filler.
- expected_change must be observably visible (a menu opens, a new dialog appears, the page scrolls, etc.).
- If the user's message is NOT a how-to / learn-by-doing request (e.g. small talk, a factual question),
  return "steps": [] and put your conversational reply in "fallback_text". Leave plan_summary empty.
""".strip()

HELP_CHECK_SYSTEM_PROMPT = """
You are watching a user complete one step of a guided walkthrough. Look at the current screenshot and decide
whether the expected change has happened yet.

Respond with JSON only, no prose:
{"done": true | false, "note": "one short phrase explaining why"}
""".strip()

ACTIVE_DONE_CHECK_SYSTEM_PROMPT = """
You are verifying whether the user's original goal has been FULLY completed on the desktop. You receive the
original goal text and the current screenshot. Be strict: only report done=true if a reasonable user would
agree the task is finished. Partial progress (a window opened, an input focused, a search submitted but
results not yet acted on) is NOT done.

Respond with JSON only, no prose:
{"done": true | false, "note": "one short phrase explaining what is or isn't done"}
""".strip()

LIVE_HELP_SYSTEM_PROMPT = """
You are Helper running in live Help mode. The user is learning how to complete a task on their Windows 11
desktop. Your job is to coach them through it one click at a time, reacting to whatever is on screen NOW.

Each turn you get the current screenshot (latest user message) and the conversation history (your past
suggestions and short notes about what happened next, including any earlier screenshots). The user clicks the
real buttons themselves; you only point with a ghost cursor and a highlight rectangle.

You MUST output ONE decision as JSON only, no prose outside the JSON. Choose exactly one shape:

A) Highlight the next single thing to click or focus:
{
  "kind": "step",
  "instruction": "Short friendly direction (one sentence).",
  "target_id": "c001",
  "target": {"x": 0-1000, "y": 0-1000, "width": 0-1000, "height": 0-1000},
  "expected_change": "What the user should see once they do it.",
  "helper_action": null
}

B) Say something without pointing (e.g., the next target is not visible yet, or you want to correct a misclick):
{
  "kind": "narrate",
  "message": "One short helpful sentence shown to the user.",
  "helper_action": null
}

C) The task is complete:
{
  "kind": "done",
  "message": "Friendly one-line confirmation."
}

helper_action (optional, runs BEFORE your message is shown) must be one of:
  {"name": "launch_app", "command": "ms-settings:", "display_name": "Settings"}
  {"name": "open_url", "url": "https://example.com"}
  {"name": "key", "keys": "ctrl+t"}
  {"name": "scroll", "direction": "down" | "up"}
Never include click_at, type_text_at, drag_and_drop, click_control. Clicking is the user's job.

Rules:
- ONE decision per turn. Do not list future steps.
- Coordinates are normalized 0-1000 over the provided screenshot.
- Visible clickable control IDs expire after each screenshot. Use only IDs from the latest Visible clickable
  controls list, never IDs remembered from earlier turns.
- If the correct visible target is in the latest Visible clickable controls list, set target_id to that exact id
  AND include the same tight target rectangle from that latest list.
- Control entries may include visible_text and automation_id. visible_text is what the user can see; automation_id is
  metadata only. Do not choose a target_id from automation_id if visible_text conflicts with your instruction.
- Only choose a target_id when the listed control's visible_text, role, or screen position clearly matches your instruction.
- If no listed control is the correct target but the target is clearly visible in the screenshot, leave target_id empty
  and provide a tight target rectangle.
- If the needed target is not visible, the listed controls disagree with the screenshot, or you are not confident which
  exact control is correct, use narrate. It is better to explain than to point at a likely-wrong spot.
- target must be a TIGHT bounding box around the EXACT clickable element. Typical size:
  width 20-120 / height 12-60 for a button or menu item; width 40-300 / height 16-40 for an input.
- If the latest history note says the user clicked elsewhere or appears stuck, prefer "narrate" to
  re-orient them gently, or pick a different target — do NOT re-emit the previous target unchanged.
- If the user's request was small talk or a factual question (not a how-to), answer with "narrate" and
  leave it at that.
- Output JSON only.
""".strip()


COMPUTER_USE_KEYWORDS = (
    "click",
    "double click",
    "right click",
    "open",
    "launch",
    "type",
    "enter",
    "press",
    "scroll",
    "drag",
    "drop",
    "hover",
    "move the mouse",
    "move my mouse",
    "select",
    "search for",
    "go to",
    "navigate",
    "close",
    "switch to",
    "focus",
    "take a screenshot",
    "look at my screen",
    "check my screen",
    "use the keyboard",
    "use the mouse",
    "on my screen",
    "on the screen",
    "on my desktop",
    "in this window",
    "for me",
    "do it",
)

CHAT_INTENT_PREFIXES = (
    "how do i",
    "how to",
    "what is",
    "what's",
    "what does",
    "why is",
    "why does",
    "why doesn't",
    "why don't",
    "explain",
    "is it",
    "is there",
    "are you",
    "do you",
    "can you explain",
    "tell me",
    "help me understand",
    "show me how",
)

_ROUTE_CACHE_MAX = 256


class GuideDispatcher(Protocol):
    def dispatch(self, action: "GuideAction") -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class GuideAction:
    name: str
    summary: str
    call_id: str | None = None
    raw_args: dict[str, Any] = field(default_factory=dict)
    normalized_x: int | None = None
    normalized_y: int | None = None
    screen_x: int | None = None
    screen_y: int | None = None
    destination_screen_x: int | None = None
    destination_screen_y: int | None = None
    text: str | None = None
    keys: str | None = None
    direction: str | None = None
    press_enter: bool | None = None
    clear_before_typing: bool | None = None
    requires_confirmation: bool = False
    confirmation_reason: str | None = None


@dataclass
class GuideTurn:
    message: str
    actions: list[GuideAction]
    done: bool
    step_index: int
    elapsed_sec: float
    capture: Capture | None


@dataclass
class GuideSession:
    goal: str
    history: HistoryManager = field(default_factory=HistoryManager)
    step_count: int = 0
    started_at: float = field(default_factory=time.monotonic)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass(frozen=True)
class HelpStep:
    instruction: str
    target_norm_x: int
    target_norm_y: int
    target_norm_width: int
    target_norm_height: int
    expected_change: str
    helper_action: dict[str, Any] | None = None

    def screen_rect(self, capture: Capture) -> tuple[int, int, int, int]:
        left_img, top_img, right_img, bottom_img = _clamped_image_rect(
            self.target_norm_x,
            self.target_norm_y,
            self.target_norm_width,
            self.target_norm_height,
            capture,
        )
        left, top = capture.to_screen_coords(left_img, top_img)
        right, bottom = capture.to_screen_coords(right_img, bottom_img)
        return left, top, max(8, right - left), max(8, bottom - top)


@dataclass(frozen=True)
class HelpPlan:
    summary: str
    steps: tuple[HelpStep, ...]
    fallback_text: str = ""
    capture: Capture | None = None

    @property
    def is_walkthrough(self) -> bool:
        return bool(self.steps)


LiveHelpKind = Literal["step", "narrate", "done"]


@dataclass(frozen=True)
class LiveHelpDecision:
    kind: LiveHelpKind
    message: str = ""
    instruction: str = ""
    expected_change: str = ""
    target_id: str = ""
    target_norm_x: int = 0
    target_norm_y: int = 0
    target_norm_width: int = 0
    target_norm_height: int = 0
    helper_action: dict[str, Any] | None = None

    @property
    def has_target_rect(self) -> bool:
        return self.target_norm_width > 0 and self.target_norm_height > 0

    def screen_rect(self, capture: Capture) -> tuple[int, int, int, int]:
        left_img, top_img, right_img, bottom_img = _clamped_image_rect(
            self.target_norm_x,
            self.target_norm_y,
            self.target_norm_width,
            self.target_norm_height,
            capture,
        )
        left, top = capture.to_screen_coords(left_img, top_img)
        right, bottom = capture.to_screen_coords(right_img, bottom_img)
        return left, top, max(8, right - left), max(8, bottom - top)

    @property
    def history_text(self) -> str:
        if self.kind == "step":
            return f"Suggested step: {self.instruction}"
        if self.kind == "done":
            return f"Walkthrough complete: {self.message}"
        return f"Note: {self.message}"


class HelpPlanError(RuntimeError):
    """Raised when the model's help-plan output cannot be parsed."""


class LimitExceeded(RuntimeError):
    """Raised when a guide session hits its step or wall-clock budget.

    Distinct from a bug so the active-mode loop can catch it and emit a
    graceful wrap-up turn instead of crashing.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _clamped_image_rect(
    norm_x: int,
    norm_y: int,
    norm_width: int,
    norm_height: int,
    capture: Capture,
) -> tuple[int, int, int, int]:
    width = max(1, capture.width)
    height = max(1, capture.height)
    left = int(norm_x / 1000 * width)
    top = int(norm_y / 1000 * height)
    right = int((norm_x + norm_width) / 1000 * width)
    bottom = int((norm_y + norm_height) / 1000 * height)
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def _with_control_candidates(
    messages: list[dict[str, Any]],
    control_candidates: list[ControlCandidate],
    capture: Capture,
) -> list[dict[str, Any]]:
    candidate_text = format_candidates_for_prompt(control_candidates, capture)
    copied = [
        {
            **message,
            "parts": [dict(part) for part in message.get("parts", [])],
        }
        for message in messages
    ]
    for message in reversed(copied):
        if message.get("role") == "user":
            message.setdefault("parts", []).append({"text": candidate_text})
            return copied
    return [
        *copied,
        {
            "role": "user",
            "parts": [{"text": candidate_text}],
        },
    ]


class HelplerAgent:
    def __init__(
        self,
        *,
        capture_provider: Callable[[], Capture] = capture_virtual_desktop,
        dispatcher: GuideDispatcher | None = None,
        client: OpenAIClient | None = None,
    ) -> None:
        self._capture_provider = capture_provider
        self._dispatcher = dispatcher
        self._client = client or make_openai_client(optional_api_key=OPENAI_API_KEY or None)
        self._route_cache: dict[str, str] = {}

    def start_guide(self, goal: str, capture: Capture | None = None) -> tuple[GuideSession, GuideTurn]:
        session = GuideSession(goal=goal)
        turn = self.step(session, user_text=goal, capture=capture)
        return session, turn

    def build_help_plan(self, goal: str, capture: Capture | None = None) -> HelpPlan:
        capture = capture or self._capture_provider()
        history = HistoryManager()
        history.add_user_turn(text=goal, screenshot=capture)
        messages = history.build_messages()
        result = self._client.chat(
            messages,
            model=REASONING_MODEL,
            system_prompt=HELP_SYSTEM_PROMPT,
        )
        return _parse_help_plan(result.text, capture)

    def plan_next_step(
        self,
        history: HistoryManager,
        *,
        control_candidates: list[ControlCandidate] | None = None,
        capture: Capture | None = None,
    ) -> LiveHelpDecision:
        """Ask the model for the SINGLE next thing the user should do.

        The caller is expected to have already pushed the latest screenshot onto
        `history` as a user turn (paired with a short outcome note) before
        calling. Returns a decision describing a target to highlight, a free-form
        narration, or completion.
        """
        messages = history.build_messages()
        if control_candidates is not None and capture is not None:
            messages = _with_control_candidates(messages, control_candidates, capture)
        result = self._client.chat(
            messages,
            model=REASONING_MODEL,
            system_prompt=LIVE_HELP_SYSTEM_PROMPT,
        )
        return _parse_live_help_decision(result.text)

    def check_step_complete(self, step: HelpStep, capture: Capture | None = None) -> tuple[bool, str]:
        capture = capture or self._capture_provider()
        history = HistoryManager()
        user_text = (
            f"Instruction: {step.instruction}\n"
            f"Expected change: {step.expected_change}\n"
            "Look at the screenshot and decide whether the expected change is now visible."
        )
        history.add_user_turn(text=user_text, screenshot=capture)
        messages = history.build_messages()
        result = self._client.chat(
            messages,
            model=REASONING_MODEL,
            system_prompt=HELP_CHECK_SYSTEM_PROMPT,
        )
        return _parse_help_check(result.text)

    def verify_goal_complete(
        self,
        session: GuideSession,
        capture: Capture | None = None,
    ) -> tuple[bool, str]:
        """Decide whether session.goal is fully complete given a fresh screenshot.

        Independent of session.history — uses an ephemeral HistoryManager with
        only the original goal + a current capture so the verifier is not
        biased by the model's own prior reasoning.
        """
        goal = (session.goal or "").strip()
        if not goal:
            return True, "no goal recorded"
        capture = capture or self._capture_provider()
        history = HistoryManager()
        user_text = (
            f"Original goal: {goal}\n"
            "Look at the screenshot and decide whether the goal is fully complete."
        )
        history.add_user_turn(text=user_text, screenshot=capture)
        messages = history.build_messages()
        result = self._client.chat(
            messages,
            model=REASONING_MODEL,
            system_prompt=ACTIVE_DONE_CHECK_SYSTEM_PROMPT,
        )
        return _parse_help_check(result.text)

    def continue_guide(
        self,
        session: GuideSession,
        capture: Capture | None = None,
        note: str | None = None,
    ) -> GuideTurn:
        message = note or self._build_continuation_note(session)
        return self.step(session, user_text=message, capture=capture, route_override="computer_use")

    @staticmethod
    def _build_continuation_note(session: GuideSession) -> str:
        goal = (session.goal or "").strip()
        goal_line = f"Original goal: {goal}\n" if goal else ""
        executed = session.history.last_executed_actions()
        summaries = [
            HelplerAgent._summarize_action(item["name"], item.get("args") or {})
            for item in executed
            if item.get("name")
        ]
        joined = " ".join(s.rstrip(".") + "." for s in summaries if s)
        if not joined:
            return (
                f"{goal_line}"
                "Continue from the latest screenshot. "
                "If the goal is fully complete, reply with text only and no tool calls."
            )
        return (
            f"{goal_line}"
            f"Helper just executed: {joined} "
            "Verify the result in the latest screenshot and decide the next step. "
            "If the goal is fully complete, reply with text only and no tool calls."
        )

    def step(
        self,
        session: GuideSession,
        *,
        user_text: str,
        capture: Capture | None = None,
        route_override: Literal["chat", "computer_use"] | None = None,
    ) -> GuideTurn:
        self._enforce_limits(session)
        fast_action = self._fast_action(user_text)
        if fast_action is not None and route_override is None:
            return self._run_fast_action(session, user_text, fast_action)

        route = route_override or self._resolve_route(user_text)
        needs_screen = route == "computer_use"
        if needs_screen:
            capture = capture or self._capture_provider()
        else:
            capture = None

        session.history.add_user_turn(text=user_text, screenshot=capture)
        model, system_prompt = self._select_generation_path(route)

        messages = session.history.build_messages()
        started = time.monotonic()
        if route == "computer_use":
            result = self._client.computer_use_step(
                messages,
                model=model,
                system_prompt=system_prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        else:
            result = self._client.chat(
                messages,
                model=model,
                system_prompt=system_prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        elapsed_sec = time.monotonic() - started

        self._record_response(session, result)
        session.step_count += 1

        actions = self._build_actions(result.tool_calls, capture)
        message = self._join_text(result.text, actions)
        turn = GuideTurn(
            message=message,
            actions=actions,
            done=not actions,
            step_index=session.step_count,
            elapsed_sec=elapsed_sec,
            capture=capture,
        )

        if self._dispatcher:
            for action in actions:
                output = self._dispatcher.dispatch(action)
                session.history.add_function_response(
                    action.name,
                    output,
                    action.call_id,
                )

        return turn

    def _run_fast_action(
        self,
        session: GuideSession,
        user_text: str,
        action: GuideAction,
    ) -> GuideTurn:
        session.history.add_user_turn(text=user_text, screenshot=None)
        session.step_count += 1

        if self._dispatcher:
            output = self._dispatcher.dispatch(action)
            session.history.add_function_response(action.name, output, action.call_id)

        message = action.summary.rstrip(".") + "."
        session.history.add_assistant_turn(message)
        return GuideTurn(
            message=message,
            actions=[action],
            done=True,
            step_index=session.step_count,
            elapsed_sec=0.0,
            capture=None,
        )

    @staticmethod
    def _route(user_text: str) -> Literal["chat", "computer_use"]:
        decision = HelplerAgent._route_rules(user_text)
        if decision == "ambiguous":
            return "computer_use"
        return decision

    @staticmethod
    def _route_rules(user_text: str) -> Literal["chat", "computer_use", "ambiguous"]:
        text = " ".join((user_text or "").strip().lower().split())
        if not text:
            return "chat"
        if text.startswith(CHAT_INTENT_PREFIXES):
            return "chat"
        has_compute_keyword = any(keyword in text for keyword in COMPUTER_USE_KEYWORDS)
        if text.endswith("?") and len(text) <= 80 and not has_compute_keyword:
            return "chat"
        if parse_fast_action(text) is not None:
            return "computer_use"
        if has_compute_keyword:
            return "computer_use"
        padded = f" {text} "
        for alias_map in (SITE_ALIASES, APP_ALIASES, HOTKEY_ALIASES):
            for key in alias_map:
                if f" {key} " in padded:
                    return "computer_use"
        return "ambiguous"

    def _resolve_route(self, user_text: str) -> Literal["chat", "computer_use"]:
        decision = self._route_rules(user_text)
        if decision != "ambiguous":
            return decision
        if not USE_ROUTE_CLASSIFIER:
            return "computer_use"
        return self._classify_with_llm(user_text)

    def _classify_with_llm(self, user_text: str) -> Literal["chat", "computer_use"]:
        key = " ".join((user_text or "").strip().lower().split())
        if not key:
            return "chat"
        cached = self._route_cache.get(key)
        if cached is not None:
            return "chat" if cached == "chat" else "computer_use"
        classifier = getattr(self._client, "classify_route", None)
        word = ""
        if classifier is not None:
            try:
                word = classifier(user_text, model=REASONING_MODEL) or ""
            except Exception as exc:
                log.warning("Route classifier raised: %s", exc)
                word = ""
        if word == "chat":
            decision: Literal["chat", "computer_use"] = "chat"
        elif word == "act":
            decision = "computer_use"
        else:
            decision = "computer_use"
        if len(self._route_cache) >= _ROUTE_CACHE_MAX:
            self._route_cache.clear()
        self._route_cache[key] = decision
        return decision

    @staticmethod
    def _fast_action(user_text: str) -> GuideAction | None:
        parsed = parse_fast_action(user_text)
        if parsed is None:
            return None
        return GuideAction(
            name=parsed.name,
            summary=parsed.summary,
            raw_args=dict(parsed.raw_args),
            keys=parsed.keys,
        )

    def _select_generation_path(self, route: Literal["chat", "computer_use"]) -> tuple[str, str]:
        if route == "computer_use":
            return AGENT_MODEL, COMPUTER_USE_SYSTEM_PROMPT
        return REASONING_MODEL, CHAT_SYSTEM_PROMPT

    def _enforce_limits(self, session: GuideSession) -> None:
        if session.step_count >= MAX_AGENT_STEPS:
            raise LimitExceeded(
                f"Guide session exceeded {MAX_AGENT_STEPS} steps.",
                kind="steps",
            )
        if time.monotonic() - session.started_at >= AGENT_TIMEOUT_SEC:
            raise LimitExceeded(
                f"Guide session exceeded {AGENT_TIMEOUT_SEC} seconds.",
                kind="timeout",
            )

    @staticmethod
    def _record_response(session: GuideSession, result: ChatResult) -> None:
        if result.text:
            session.history.add_assistant_turn(result.text)
        for call in result.tool_calls:
            session.history.add_function_call(call.name, call.arguments, call.call_id)

    @classmethod
    def _join_text(cls, text: str, actions: list[GuideAction]) -> str:
        text = (text or "").strip()
        if text:
            return text
        return cls._fallback_message(actions)

    @staticmethod
    def _fallback_message(actions: list[GuideAction]) -> str:
        if not actions:
            return "(no reply)"
        summaries = [
            HelplerAgent._lower_first(action.summary.rstrip("."))
            for action in actions[:2]
            if action.summary
        ]
        fallback = ", then ".join(summaries)
        if not fallback:
            return "Working on it."
        return f"I'll {fallback}."

    @staticmethod
    def _lower_first(text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]

    def _build_actions(self, tool_calls: list[ToolCall], capture: Capture | None) -> list[GuideAction]:
        return [
            self._action_from_call(call.name, call.arguments, capture, call.call_id)
            for call in tool_calls
        ]

    def _action_from_call(
        self,
        name: str,
        args: dict[str, Any],
        capture: Capture | None,
        call_id: str | None,
    ) -> GuideAction:
        safety_decision = args.get("safety_decision") or {}
        requires_confirmation = safety_decision.get("decision") == "require_confirmation"
        confirmation_reason = safety_decision.get("explanation")

        point = self._screen_point(capture, args.get("x"), args.get("y"))
        destination = self._screen_point(capture, args.get("destination_x"), args.get("destination_y"))

        summary = self._summarize_action(name, args)
        return GuideAction(
            name=name,
            summary=summary,
            call_id=call_id,
            raw_args=args,
            normalized_x=self._maybe_normalized(args.get("x")),
            normalized_y=self._maybe_normalized(args.get("y")),
            screen_x=point[0] if point else None,
            screen_y=point[1] if point else None,
            destination_screen_x=destination[0] if destination else None,
            destination_screen_y=destination[1] if destination else None,
            text=args.get("text"),
            keys=args.get("keys"),
            direction=args.get("direction"),
            press_enter=args.get("press_enter"),
            clear_before_typing=args.get("clear_before_typing"),
            requires_confirmation=requires_confirmation,
            confirmation_reason=confirmation_reason,
        )

    @staticmethod
    def _maybe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _maybe_normalized(cls, value: Any) -> int | None:
        parsed = cls._maybe_int(value)
        if parsed is None:
            return None
        return max(0, min(1000, parsed))

    def _screen_point(self, capture: Capture | None, x: Any, y: Any) -> tuple[int, int] | None:
        if capture is None:
            return None
        norm_x = self._maybe_int(x)
        norm_y = self._maybe_int(y)
        if norm_x is None or norm_y is None:
            return None
        norm_x = max(0, min(1000, norm_x))
        norm_y = max(0, min(1000, norm_y))
        image_x = max(0, min(capture.width - 1, int((norm_x / 1000) * capture.width)))
        image_y = max(0, min(capture.height - 1, int((norm_y / 1000) * capture.height)))
        return capture.to_screen_coords(image_x, image_y)

    @staticmethod
    def _summarize_action(name: str, args: dict[str, Any]) -> str:
        fixed_summaries = {
            "open_web_browser": "Open the web browser.",
            "wait_5_seconds": "Wait five seconds for the UI to settle.",
            "go_back": "Go back to the previous page or view.",
            "go_forward": "Go forward to the next page or view.",
            "click_at": "Click the highlighted location.",
            "double_click_at": "Double-click the highlighted location.",
            "right_click_at": "Right-click the highlighted location.",
            "hover_at": "Hover over the highlighted location.",
            "drag_and_drop": "Drag from the highlighted starting point to the highlighted destination.",
        }
        if name in fixed_summaries:
            return fixed_summaries[name]
        if name == "navigate":
            return f"Navigate to {safe_target_label(args.get('url', 'the requested URL'))}."
        if name == "type_text_at":
            text = str(args.get("text", ""))
            return f"Type {len(text)} characters at the highlighted location."
        if name == "key_combination":
            return f"Press {args.get('keys', 'the requested key combination')}."
        if name == "scroll_document":
            return f"Scroll the document {args.get('direction', 'down')}."
        if name == "scroll_at":
            return f"Scroll {args.get('direction', 'down')} at the highlighted location."
        return f"Perform '{name}'."


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ALLOWED_HELPER_ACTIONS = {"launch_app", "open_url", "key", "scroll"}


def _parse_help_plan(text: str, capture: Capture) -> HelpPlan:
    payload = _extract_json_object(text)
    if payload is None:
        raise HelpPlanError("Help plan response was not valid JSON.")

    summary = str(payload.get("plan_summary") or "").strip()
    fallback_text = str(payload.get("fallback_text") or "").strip()
    raw_steps = payload.get("steps")
    steps: list[HelpStep] = []
    if isinstance(raw_steps, list):
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            step = _parse_help_step(raw)
            if step is not None:
                steps.append(step)
    return HelpPlan(
        summary=summary,
        steps=tuple(steps),
        fallback_text=fallback_text,
        capture=capture,
    )


def _parse_help_step(raw: dict[str, Any]) -> HelpStep | None:
    instruction = str(raw.get("instruction") or "").strip()
    if not instruction:
        return None
    target = raw.get("target") or {}
    if not isinstance(target, dict):
        return None
    norm_x = _coerce_norm(target.get("x"))
    norm_y = _coerce_norm(target.get("y"))
    norm_w = _coerce_norm(target.get("width"))
    norm_h = _coerce_norm(target.get("height"))
    if None in (norm_x, norm_y, norm_w, norm_h):
        return None
    if not _valid_norm_rect(norm_x, norm_y, norm_w, norm_h):
        return None
    if norm_w * norm_h > 400_000:
        log.warning(
            "Help step target rect is suspiciously large (%dx%d normalized): %s",
            norm_w, norm_h, instruction,
        )
    expected_change = str(raw.get("expected_change") or "").strip()
    helper_action = _sanitize_helper_action(raw.get("helper_action"))
    return HelpStep(
        instruction=instruction,
        target_norm_x=norm_x,
        target_norm_y=norm_y,
        target_norm_width=norm_w,
        target_norm_height=norm_h,
        expected_change=expected_change,
        helper_action=helper_action,
    )


def _sanitize_helper_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or "").strip().lower()
    if name not in _ALLOWED_HELPER_ACTIONS:
        return None
    sanitized: dict[str, Any] = {"name": name}
    for key, val in value.items():
        if key == "name":
            continue
        if isinstance(val, (str, int, float, bool)):
            sanitized[key] = val
    return sanitized


def _parse_live_help_decision(text: str) -> LiveHelpDecision:
    """Parse a single live-help decision. Falls back to narrate on any error."""
    payload = _extract_json_object(text)
    if payload is None:
        return LiveHelpDecision(kind="narrate", message="I'm not sure what to do next. Try rephrasing.")

    kind = str(payload.get("kind") or "").strip().lower()
    helper_action = _sanitize_helper_action(payload.get("helper_action"))

    if kind == "done":
        message = str(payload.get("message") or "").strip() or "Walkthrough complete."
        return LiveHelpDecision(kind="done", message=message, helper_action=helper_action)

    if kind == "narrate":
        message = str(payload.get("message") or "").strip()
        if not message:
            return LiveHelpDecision(kind="narrate", message="(no reply)")
        return LiveHelpDecision(kind="narrate", message=message, helper_action=helper_action)

    if kind == "step":
        instruction = str(payload.get("instruction") or "").strip()
        target_id = str(payload.get("target_id") or "").strip()
        target = payload.get("target") or {}
        if not instruction:
            fallback = instruction or "Take a look at the screen."
            return LiveHelpDecision(kind="narrate", message=fallback, helper_action=helper_action)
        norm_x = norm_y = norm_w = norm_h = None
        if isinstance(target, dict):
            norm_x = _coerce_norm(target.get("x"))
            norm_y = _coerce_norm(target.get("y"))
            norm_w = _coerce_norm(target.get("width"))
            norm_h = _coerce_norm(target.get("height"))
        has_rect = None not in (norm_x, norm_y, norm_w, norm_h)
        if has_rect and not _valid_norm_rect(norm_x, norm_y, norm_w, norm_h):
            has_rect = False
        if not target_id and not has_rect:
            return LiveHelpDecision(kind="narrate", message=instruction, helper_action=helper_action)
        if not has_rect:
            norm_x = norm_y = norm_w = norm_h = 0
        return LiveHelpDecision(
            kind="step",
            instruction=instruction,
            expected_change=str(payload.get("expected_change") or "").strip(),
            target_id=target_id,
            target_norm_x=norm_x,
            target_norm_y=norm_y,
            target_norm_width=norm_w,
            target_norm_height=norm_h,
            helper_action=helper_action,
        )

    fallback = str(payload.get("message") or payload.get("instruction") or "").strip()
    if fallback:
        return LiveHelpDecision(kind="narrate", message=fallback, helper_action=helper_action)
    return LiveHelpDecision(kind="narrate", message="I'm not sure what to do next.")


def _parse_help_check(text: str) -> tuple[bool, str]:
    payload = _extract_json_object(text)
    if payload is None:
        return False, "could not parse model reply"
    done = bool(payload.get("done"))
    note = str(payload.get("note") or "").strip()
    return done, note


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(candidate)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_norm(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = int(float(value))
        except (TypeError, ValueError, OverflowError):
            return None
    if not 0 <= parsed <= 1000:
        return None
    return parsed


def _valid_norm_rect(x: int, y: int, width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    return x + width <= 1000 and y + height <= 1000


def _print_turn(turn: GuideTurn) -> None:
    payload = {
        "message": turn.message,
        "done": turn.done,
        "step_index": turn.step_index,
        "elapsed_sec": round(turn.elapsed_sec, 3),
        "actions": [asdict(action) for action in turn.actions],
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Helper guide-mode planning step.")
    parser.add_argument("goal", help="The user's desktop task.")
    args = parser.parse_args()

    agent = HelplerAgent()
    _, turn = agent.start_guide(args.goal)
    _print_turn(turn)


if __name__ == "__main__":
    main()
