from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from threading import Event, Thread
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal
from PIL import Image, ImageChops, ImageStat

from control_inventory import (
    ADD_ACTION_WORDS,
    BARE_EXTENDED_ACTION_LABEL_WORDS,
    CLEAR_CLOSE_WORDS,
    CONFIRM_ACTION_WORDS,
    ControlCandidate,
    EXCLUSIVE_ACTION_FAMILIES,
    PAY_ACTION_WORDS,
    SEARCH_ACTION_WORDS,
    TargetResolution,
    collect_control_candidates,
    resolve_candidate_target,
    snap_candidate_target,
)
from help_intents import (
    tokenize_control as _tokenize_control,
    tokenize_instruction as _tokenize_instruction,
    tokens_from_text as _tokens_from_text,
)
from history import HistoryManager
from launch_safety import HELP_LAUNCH_COMMANDS, LaunchError, launch_target, safe_target_label
from ocr_text import (
    OcrTextProvider,
    OcrTextVerification,
    default_ocr_text_provider,
    expected_text_evidence_for_target,
    verify_target_text,
)
from rect_snap import SnapResult, snap_to_control
from screen import capture_active_monitor
from target_quality import TargetQuality, evaluate_target_quality

if TYPE_CHECKING:
    from agent import HelplerAgent, LiveHelpDecision
    from computer_control import ComputerController
    from screen import Capture

CandidateProvider = Callable[["Capture"], list[ControlCandidate]]
CaptureProvider = Callable[[], "Capture"]
Snapper = Callable[[tuple[int, int, int, int], str], SnapResult]
OverlayClearBarrier = Callable[[], None]

log = logging.getLogger("helper.help_session")

IDLE_RECHECK_SEC = 5.0
MAX_TURNS = 25
POST_ACTION_SETTLE_SEC = 0.6
POST_CLICK_SETTLE_SEC = 0.35
OVERLAY_CLEAR_SETTLE_SEC = 0.05
CLICK_HIT_MARGIN_PX = 24
SURFACE_PROMOTION_CONTROL_TYPES = frozenset({"group", "headeritem", "list", "menu", "pane", "toolbar", "window"})
SURFACE_REVALIDATION_CONTROL_TYPES = frozenset(
    {"datagrid", "grid", "group", "list", "menu", "pane", "table", "toolbar", "window"}
)
CANDIDATE_EMPTY_RETRIES = 2
CANDIDATE_EMPTY_RETRY_SEC = 0.08
UIA_BACKED_TARGET_SOURCES = frozenset(
    {"target_id", "text_match", "candidate_snap", "snap"}
)
MIN_REVALIDATION_OVERLAP_FRACTION = 0.25
FOREGROUND_COVERAGE_MIN_FRACTION = 0.55
FOREGROUND_CENTER_COVERAGE_MIN_FRACTION = 0.20
SAME_RANK_PREVIOUS_OWNER_MIN_AREA_RATIO = 2.5
SAME_RANK_COVERING_CONTROL_TYPES = frozenset(
    {"button", "hyperlink", "menuitem", "option", "splitbutton", "tabitem"}
)
SAME_RANK_OWNING_COVERING_CONTROL_TYPES = frozenset(
    {
        "cell",
        "checkbox",
        "combobox",
        "dataitem",
        "datagridcell",
        "edit",
        "gridcell",
        "headeritem",
        "listitem",
        "radiobutton",
        "row",
        "slider",
        "spinner",
        "tableitem",
        "treeitem",
    }
)
SAME_RANK_TRANSIENT_SURFACE_CONTROL_TYPES = frozenset({"group", "list", "menu", "pane", "toolbar", "window"})
SAME_RANK_TRANSIENT_SURFACE_WORDS = frozenset(
    {"dialog", "down", "dropdown", "flyout", "menu", "modal", "popup", "popover", "suggestion", "suggestions", "toast"}
)
GENERIC_ACTION_REVALIDATION_CONTEXT_MARGIN_PX = 96
GENERIC_ACTION_REVALIDATION_CONTEXT_DIFF_FLOOR = 0.010
EPHEMERAL_ACTION_REVALIDATION_CONTEXT_DIFF_FLOOR = 0.005
ROW_REVALIDATION_CONTROL_TYPES = frozenset(
    {"dataitem", "listitem", "row", "tableitem", "treeitem"}
)
ROW_REVALIDATION_GENERIC_WORDS = frozenset(
    {"card", "dataitem", "item", "listitem", "record", "row", "tableitem", "treeitem"}
)
TABULAR_REVALIDATION_CONTROL_TYPES = frozenset(
    {"cell", "datagridcell", "gridcell", "headeritem", "rowheader"}
)
TABULAR_REVALIDATION_GENERIC_WORDS = frozenset(
    {
        "cell",
        "column",
        "datagridcell",
        "gridcell",
        "header",
        "headeritem",
        "heading",
        "rowheader",
        "table",
    }
)
ACTION_REVALIDATION_CONTROL_TYPES = frozenset(
    {"button", "hyperlink", "menuitem", "splitbutton", "tabitem"}
)
CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES = frozenset(
    {"checkbox", "combobox", "edit", "radiobutton", "slider", "spinner"}
)
ACTION_CONTEXT_REVALIDATION_CONTROL_TYPES = (
    ROW_REVALIDATION_CONTROL_TYPES
    | frozenset({"group", "list", "menu", "pane", "toolbar", "window"})
)
ACTION_CONTEXT_TEXT_CONTROL_TYPES = frozenset({"label", "statictext", "text"})
ACTION_CONTEXT_REVALIDATION_GENERIC_WORDS = ROW_REVALIDATION_GENERIC_WORDS | frozenset(
    {
        "account",
        "accounts",
        "body",
        "button",
        "caption",
        "display",
        "displaying",
        "label",
        "labels",
        "menu",
        "pane",
        "panel",
        "people",
        "person",
        "persons",
        "profile",
        "profiles",
        "request",
        "requests",
        "show",
        "showing",
        "statictext",
        "text",
        "title",
        "user",
        "users",
    }
)
WEAK_ACTION_CONTEXT_REVALIDATION_WORDS = frozenset(
    {
        "dialog",
        "dialogs",
        "modal",
        "modals",
        "pane",
        "panes",
        "panel",
        "panels",
        "popup",
        "popups",
        "window",
        "windows",
    }
)
ACTION_IDENTITY_REVALIDATION_GENERIC_WORDS = frozenset(
    {
        "action",
        "actions",
        "button",
        "buttons",
        "control",
        "controls",
        "hyperlink",
        "item",
        "link",
        "links",
        "menu",
        "menuitem",
        "split",
        "splitbutton",
        "tab",
        "tabitem",
    }
)
CONTROL_IDENTITY_REVALIDATION_GENERIC_WORDS = ACTION_IDENTITY_REVALIDATION_GENERIC_WORDS | frozenset(
    {
        "check",
        "checkbox",
        "combo",
        "combobox",
        "dropdown",
        "edit",
        "field",
        "input",
        "option",
        "radio",
        "radiobutton",
        "select",
        "selected",
        "selection",
        "slider",
        "spinner",
        "switch",
        "text",
        "textbox",
        "toggle",
        "value",
    }
)
CONTROL_CONTEXT_REVALIDATION_GENERIC_WORDS = CONTROL_IDENTITY_REVALIDATION_GENERIC_WORDS | frozenset(
    {
        "checked",
        "dialog",
        "disable",
        "disabled",
        "address",
        "enable",
        "enabled",
        "email",
        "envelope",
        "find",
        "location",
        "mail",
        "modal",
        "off",
        "on",
        "preferences",
        "search",
        "settings",
        "unchecked",
        "url",
        "window",
    }
)
CONTROL_CONTEXT_LABEL_TYPES = frozenset({"headeritem", "label", "statictext", "text"})
CONTROL_CONTEXT_SECTION_TYPES = frozenset({"group", "list", "pane", "window"})
CONTROL_CONTEXT_ROW_TYPES = frozenset({"dataitem", "listitem", "row", "tableitem", "treeitem"})
SURFACE_IDENTITY_REVALIDATION_GENERIC_WORDS = frozenset(
    {
        "area",
        "card",
        "container",
        "datagrid",
        "dialog",
        "grid",
        "group",
        "list",
        "menu",
        "modal",
        "pane",
        "panel",
        "popup",
        "preferences",
        "section",
        "settings",
        "surface",
        "table",
        "toolbar",
        "window",
    }
)
WINDOW_TITLE_REVALIDATION_GENERIC_WORDS = frozenset(
    {
        "app",
        "application",
        "brave",
        "browser",
        "chrome",
        "chromium",
        "dialog",
        "edge",
        "firefox",
        "foreground",
        "google",
        "microsoft",
        "modal",
        "page",
        "popup",
        "safari",
        "tab",
        "window",
    }
)
GENERIC_ACTION_REVALIDATION_WORDS = (
    CONFIRM_ACTION_WORDS
    | CLEAR_CLOSE_WORDS
    | ADD_ACTION_WORDS
    | PAY_ACTION_WORDS
    | set().union(*EXCLUSIVE_ACTION_FAMILIES)
)

OVERSIZED_AREA_THRESHOLD = 100_000
OVERSIZED_EDGE_THRESHOLD = 400
RAW_SNAP_EXCLUSIVE_ACTION_FAMILIES = (
    CONFIRM_ACTION_WORDS,
    CLEAR_CLOSE_WORDS,
    ADD_ACTION_WORDS,
    PAY_ACTION_WORDS,
    *EXCLUSIVE_ACTION_FAMILIES,
)
RAW_SNAP_LABEL_GENERIC_WORDS = frozenset(
    {
        "a",
        "an",
        "area",
        "button",
        "cell",
        "checkbox",
        "choose",
        "combobox",
        "click",
        "control",
        "dropdown",
        "field",
        "find",
        "focus",
        "go",
        "here",
        "highlighted",
        "hit",
        "icon",
        "item",
        "list",
        "menu",
        "navigate",
        "open",
        "option",
        "press",
        "radio",
        "radiobutton",
        "row",
        "select",
        "tab",
        "tap",
        "the",
        "this",
        "visit",
    }
)


def looks_oversized(decision: "LiveHelpDecision") -> bool:
    """A target rect in normalized 0-1000 space that is panel-sized rather
    than a tight control bounding box. The model occasionally returns a
    whole-panel box when it can't localize precisely; better to show no
    rectangle than a wrong one.
    """
    w = decision.target_norm_width
    h = decision.target_norm_height
    return (w * h) > OVERSIZED_AREA_THRESHOLD or max(w, h) > OVERSIZED_EDGE_THRESHOLD


def click_hit_margin(rect: tuple[int, int, int, int]) -> int:
    _x, _y, width, height = rect
    shortest_edge = max(1, min(abs(int(width)), abs(int(height))))
    return max(4, min(CLICK_HIT_MARGIN_PX, shortest_edge // 3))


def build_target_diagnostic(
    *,
    decision: "LiveHelpDecision",
    capture: "Capture",
    candidates: list[ControlCandidate],
    target: TargetResolution,
    quality: TargetQuality | None = None,
    ocr: OcrTextVerification | None = None,
    overlay_rect: tuple[int, int, int, int] | None = None,
    rejected_reason: str = "",
) -> dict[str, Any]:
    model_rect = decision.screen_rect(capture) if decision.has_target_rect else None
    return {
        "capture": {
            "width": capture.width,
            "height": capture.height,
            "monitor_left": capture.monitor_left,
            "monitor_top": capture.monitor_top,
            "scale": capture.scale,
        },
        "model": {
            "kind": decision.kind,
            "instruction": decision.instruction,
            "expected_change": decision.expected_change,
            "target_id": decision.target_id,
            "target_norm": {
                "x": decision.target_norm_x,
                "y": decision.target_norm_y,
                "width": decision.target_norm_width,
                "height": decision.target_norm_height,
            },
            "screen_rect": model_rect,
        },
        "resolution": {
            "rect": target.rect,
            "confidence": round(float(target.confidence), 4),
            "source": target.source,
            "matched_text": target.matched_text,
            "target_id": target.target_id,
            "rejected_reason": target.rejected_reason,
        },
        "quality": _quality_payload(quality),
        "ocr": _ocr_payload(ocr),
        "overlay": {
            "emitted": overlay_rect is not None,
            "rect": overlay_rect,
            "rejected_reason": rejected_reason or target.rejected_reason,
        },
        "candidates": [
            {
                "id": candidate.id,
                "text": candidate.text,
                "control_type": candidate.control_type,
                "rect": candidate.rect,
                "automation_id": candidate.automation_id,
                "window_rank": candidate.window_rank,
            }
            for candidate in candidates[:20]
        ],
        "candidate_count": len(candidates),
    }


def _quality_payload(quality: TargetQuality | None) -> dict[str, Any] | None:
    if quality is None:
        return None
    return {
        "accepted": quality.accepted,
        "reason": quality.reason,
        "visible_fraction": round(float(quality.visible_fraction), 4),
        "visual_activity": round(float(quality.visual_activity), 4),
        "boundary_activity": round(float(quality.boundary_activity), 4),
        "target_area_fraction": round(float(quality.target_area_fraction), 4),
    }


def _ocr_payload(ocr: OcrTextVerification | None) -> dict[str, Any] | None:
    if ocr is None:
        return None
    return {
        "accepted": ocr.accepted,
        "reason": ocr.reason,
        "expected_text": ocr.expected_text,
        "recognized_text": ocr.recognized_text,
        "available": ocr.available,
        "error": ocr.error,
        "elapsed_ms": round(float(ocr.elapsed_ms), 2),
    }


def target_control_type_for_resolution(
    target: TargetResolution,
    candidates: list[ControlCandidate],
) -> str:
    if target.target_id:
        candidate = next((item for item in candidates if item.id == target.target_id), None)
        if candidate is not None:
            return candidate.control_type
    matches = [
        candidate
        for candidate in candidates
        if _rect_iou(candidate.rect, target.rect) >= 0.80
    ]
    if len(matches) == 1:
        return matches[0].control_type
    return ""


def resolve_help_target(
    decision: "LiveHelpDecision",
    capture: "Capture",
    candidates: list[ControlCandidate],
    *,
    snapper: Snapper = snap_to_control,
    clip_to_capture: bool = True,
) -> TargetResolution:
    model_rect = decision.screen_rect(capture) if decision.has_target_rect else None

    target = resolve_candidate_target(
        target_id=decision.target_id,
        instruction=decision.instruction,
        candidates=candidates,
        model_rect=model_rect,
    )
    if target is not None and not target.rejected_reason:
        return _maybe_clip_resolution_to_capture(target, capture, clip_to_capture)
    if target is not None:
        log.info(
            "Ignoring invalid target_id=%s for instruction=%r: %s",
            decision.target_id,
            decision.instruction,
            target.rejected_reason,
        )

    if decision.target_id:
        if (
            target is not None
            and target.rejected_reason == "unknown target_id"
            and model_rect is None
        ):
            return target
        text_target = resolve_candidate_target(
            target_id="",
            instruction=decision.instruction,
            candidates=candidates,
            model_rect=model_rect,
        )
        if (
            target is not None
            and target.rejected_reason
            and model_rect is not None
            and _target_id_is_surface_candidate(decision.target_id, candidates)
            and _instruction_has_surface_promotion_context(decision.instruction)
        ):
            candidate_snap = snap_candidate_target(
                instruction=decision.instruction,
                candidates=candidates,
                model_rect=model_rect,
            )
            if candidate_snap is not None and not candidate_snap.rejected_reason:
                return _maybe_clip_resolution_to_capture(candidate_snap, capture, clip_to_capture)
        if text_target is not None and not text_target.rejected_reason:
            if (
                target is not None
                and target.rejected_reason == "target_id ambiguous"
                and text_target.target_id == target.target_id
            ):
                if (
                    _instruction_has_dialog_resolution_context(decision.instruction)
                    and _target_has_dialog_resolution_evidence(text_target.target_id, candidates)
                ):
                    return _maybe_clip_resolution_to_capture(
                        text_target,
                        capture,
                        clip_to_capture,
                    )
                return target
            return _maybe_clip_resolution_to_capture(text_target, capture, clip_to_capture)

        if target is not None and target.rejected_reason == "target_id semantic mismatch":
            if (
                text_target is not None
                and text_target.rejected_reason == "ambiguous text match"
                and _ambiguous_text_match_has_duplicate_visible_label(text_target, candidates)
            ):
                return text_target
            if model_rect is not None:
                candidate_snap = snap_candidate_target(
                    instruction=decision.instruction,
                    candidates=candidates,
                    model_rect=model_rect,
                )
                if candidate_snap is not None and not candidate_snap.rejected_reason:
                    return _maybe_clip_resolution_to_capture(candidate_snap, capture, clip_to_capture)
                if text_target is not None and not text_target.rejected_reason:
                    return _maybe_clip_resolution_to_capture(text_target, capture, clip_to_capture)
                if text_target is not None and text_target.rejected_reason == "ambiguous text match":
                    return text_target
            return target

        if model_rect is not None:
            candidate_snap = snap_candidate_target(
                instruction=decision.instruction,
                candidates=candidates,
                model_rect=model_rect,
            )
            if candidate_snap is not None and not candidate_snap.rejected_reason:
                if (
                    target is not None
                    and target.rejected_reason == "target_id ambiguous"
                    and target.target_id
                    and candidate_snap.target_id == target.target_id
                ):
                    if not _instruction_has_dialog_resolution_context(decision.instruction):
                        return target
                    return _maybe_clip_resolution_to_capture(
                        candidate_snap,
                        capture,
                        clip_to_capture,
                    )
                return _maybe_clip_resolution_to_capture(candidate_snap, capture, clip_to_capture)
            if candidate_snap is not None:
                if (
                    target is not None
                    and target.rejected_reason == "target_id ambiguous"
                    and target.target_id
                    and candidate_snap.target_id == target.target_id
                    and candidate_snap.rejected_reason == "ambiguous candidate snap"
                    and _target_id_has_foreground_exact_text_duplicate(target.target_id, candidates)
                    and not _instruction_has_dialog_resolution_context(decision.instruction)
                ):
                    return target
                return candidate_snap

        if target is not None and target.rejected_reason:
            return target

    if model_rect is None:
        return TargetResolution(
            rect=(0, 0, 0, 0),
            confidence=0.0,
            source="none",
            rejected_reason="no resolvable target",
        )

    if target is not None and target.rejected_reason == "ambiguous text match":
        return target

    candidate_snap = snap_candidate_target(
        instruction=decision.instruction,
        candidates=candidates,
        model_rect=model_rect,
    )
    if candidate_snap is not None and not candidate_snap.rejected_reason:
        return _maybe_clip_resolution_to_capture(candidate_snap, capture, clip_to_capture)
    if candidate_snap is not None:
        return candidate_snap
    if candidates:
        return TargetResolution(
            rect=model_rect,
            confidence=0.0,
            source="candidate_snap",
            rejected_reason="candidate snapshot no match",
        )

    try:
        snap = snapper(model_rect, decision.instruction)
    except Exception:
        log.exception("snap_to_control raised")
        snap = SnapResult(rect=model_rect, confidence=0.0, source="model")

    if snap.rejected_reason:
        return TargetResolution(
            rect=snap.rect,
            confidence=snap.confidence,
            source="snap",
            matched_text=snap.matched_text,
            rejected_reason=snap.rejected_reason,
        )

    if snap.source == "uia":
        if _raw_snap_action_mismatch(
            decision.instruction,
            snap.matched_text,
        ) or _raw_snap_label_mismatch(decision.instruction, snap.matched_text):
            return TargetResolution(
                rect=snap.rect,
                confidence=snap.confidence,
                source="snap",
                matched_text=snap.matched_text,
                rejected_reason="candidate semantic mismatch",
            )
        if candidates:
            return TargetResolution(
                rect=snap.rect,
                confidence=snap.confidence,
                source="snap",
                matched_text=snap.matched_text,
                rejected_reason="fresh snap inconsistent with candidate snapshot",
            )
        return _maybe_clip_resolution_to_capture(TargetResolution(
            rect=snap.rect,
            confidence=snap.confidence,
            source="snap",
            matched_text=snap.matched_text,
        ), capture, clip_to_capture)

    if looks_oversized(decision):
        return TargetResolution(
            rect=model_rect,
            confidence=snap.confidence,
            source="model",
            matched_text=snap.matched_text,
            rejected_reason="oversized target",
        )

    return _maybe_clip_resolution_to_capture(TargetResolution(
        rect=model_rect,
        confidence=snap.confidence,
        source="model",
        matched_text=snap.matched_text,
    ), capture, clip_to_capture)


def _raw_snap_action_mismatch(instruction: str, matched_text: str) -> bool:
    instruction_tokens = _tokens_from_text(instruction) | _tokenize_instruction(instruction)
    control_tokens = _tokens_from_text(matched_text) | _tokenize_control(matched_text)
    if not instruction_tokens or not control_tokens:
        return False
    requested_indexes = [
        index
        for index, family in enumerate(RAW_SNAP_EXCLUSIVE_ACTION_FAMILIES)
        if instruction_tokens & family
    ]
    if not requested_indexes:
        return False
    matched_indexes = [
        index
        for index, family in enumerate(RAW_SNAP_EXCLUSIVE_ACTION_FAMILIES)
        if control_tokens & family
    ]
    if not matched_indexes:
        return False
    return not bool(set(requested_indexes) & set(matched_indexes))


def _raw_snap_label_mismatch(instruction: str, matched_text: str) -> bool:
    if _raw_snap_bare_search_filter_mismatch(instruction, matched_text):
        return True
    if _raw_snap_bare_extended_label_mismatch(instruction, matched_text):
        return True
    instruction_tokens = (
        _tokens_from_text(instruction) | _tokenize_instruction(instruction)
    ) - RAW_SNAP_LABEL_GENERIC_WORDS
    control_tokens = (
        _tokens_from_text(matched_text) | _tokenize_control(matched_text)
    ) - RAW_SNAP_LABEL_GENERIC_WORDS
    if not instruction_tokens or not control_tokens:
        return False
    return not bool(instruction_tokens & control_tokens)


def _raw_snap_bare_search_filter_mismatch(instruction: str, matched_text: str) -> bool:
    request_words = _raw_snap_literal_label_words(instruction)
    if len(request_words) != 1 or request_words[0] not in SEARCH_ACTION_WORDS:
        return False
    matched_words = set(_raw_snap_literal_label_words(matched_text))
    return request_words[0] in matched_words and bool(matched_words & {"filter", "filters"})


def _raw_snap_bare_extended_label_mismatch(instruction: str, matched_text: str) -> bool:
    request_words = _raw_snap_literal_label_words(instruction)
    if len(request_words) != 1:
        return False
    matched_words = _raw_snap_literal_label_words(matched_text)
    if len(matched_words) <= 1:
        return False
    requested = request_words[0]
    if requested not in BARE_EXTENDED_ACTION_LABEL_WORDS:
        return False
    if requested not in matched_words:
        return False
    return bool(_raw_snap_bare_extended_label_extra_words(set(matched_words), set(request_words)))


def _raw_snap_bare_extended_label_extra_words(
    matched_words: set[str],
    request_words: set[str],
) -> set[str]:
    extra_words = matched_words - request_words - {
        "alt",
        "cmd",
        "command",
        "control",
        "ctrl",
        "keyboard",
        "meta",
        "option",
        "shortcut",
        "shift",
        "win",
        "windows",
    }
    if matched_words & {
        "alt",
        "cmd",
        "command",
        "control",
        "ctrl",
        "meta",
        "option",
        "shift",
        "win",
        "windows",
    }:
        extra_words = {word for word in extra_words if len(word) > 1}
    return extra_words


def _raw_snap_literal_label_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    excluded = {
        "a",
        "an",
        "at",
        "button",
        "choose",
        "click",
        "control",
        "focus",
        "hit",
        "now",
        "please",
        "press",
        "select",
        "tap",
        "the",
        "this",
        "that",
        "use",
    }
    return [word for word in words if word not in excluded]


def _instruction_has_dialog_resolution_context(instruction: str) -> bool:
    return bool(re.search(r"\b(?:dialog|modal|popup)\b", instruction or "", re.IGNORECASE))


def _target_has_dialog_resolution_evidence(
    target_id: str,
    candidates: list[ControlCandidate],
) -> bool:
    if not target_id:
        return False
    target_candidate = next((item for item in candidates if item.id == target_id), None)
    if target_candidate is None:
        return False
    dialog_tokens = {"dialog", "modal", "popup"}
    target_tokens = _surface_evidence_tokens(target_candidate)
    # A stale lower-rank duplicate can carry dialog words only through its own
    # window title; require stronger containing-surface evidence in that case.
    if (
        target_tokens & dialog_tokens
        and not _target_id_has_foreground_exact_text_duplicate(target_id, candidates)
    ):
        return True
    for candidate in candidates:
        if candidate.id == target_candidate.id:
            continue
        if candidate.control_type not in {"group", "pane", "window"}:
            continue
        if not _same_revalidation_context_window(target_candidate, candidate):
            continue
        if not _rect_contains(_expand_rect(candidate.rect, 4), target_candidate.rect):
            continue
        if _surface_evidence_tokens(candidate) & dialog_tokens:
            return True
    return False


def _surface_evidence_tokens(candidate: ControlCandidate) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                (
                    candidate.text or "",
                    candidate.control_type or "",
                    candidate.automation_id or "",
                    candidate.window_title or "",
                )
            ).lower(),
        )
    )


def _expand_rect(
    rect: tuple[int, int, int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    return (x - margin, y - margin, width + margin * 2, height + margin * 2)


def _rect_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh


def _instruction_has_surface_promotion_context(instruction: str) -> bool:
    return bool(
        re.search(
            r"\b(?:in|inside|on|within)\s+(?:the\s+)?"
            r"(?:group|header|list|menu|pane|panel|toolbar|window)\b",
            instruction or "",
            re.IGNORECASE,
        )
    )


def _target_id_is_surface_candidate(
    target_id: str,
    candidates: list[ControlCandidate],
) -> bool:
    if not target_id:
        return False
    return any(
        candidate.id == target_id and candidate.control_type in SURFACE_PROMOTION_CONTROL_TYPES
        for candidate in candidates
    )


def _target_id_has_foreground_exact_text_duplicate(
    target_id: str,
    candidates: list[ControlCandidate],
) -> bool:
    selected = next((candidate for candidate in candidates if candidate.id == target_id), None)
    if selected is None or selected.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    selected_words = _tokens_from_text(selected.text)
    if not selected_words:
        return False
    for candidate in candidates:
        if candidate.id == selected.id:
            continue
        if candidate.control_type != selected.control_type:
            continue
        if candidate.window_rank >= selected.window_rank:
            continue
        if _tokens_from_text(candidate.text) == selected_words:
            return True
    return False


def _ambiguous_text_match_has_duplicate_visible_label(
    target: TargetResolution,
    candidates: list[ControlCandidate],
) -> bool:
    if target.rejected_reason != "ambiguous text match" or not target.target_id:
        return False
    selected = next((candidate for candidate in candidates if candidate.id == target.target_id), None)
    if selected is None:
        return False
    selected_text = selected.text.strip().casefold()
    if not selected_text:
        return False
    for candidate in candidates:
        if candidate.id == selected.id:
            continue
        if candidate.control_type != selected.control_type:
            continue
        if candidate.text.strip().casefold() == selected_text:
            return True
    return False


def _maybe_clip_resolution_to_capture(
    target: TargetResolution,
    capture: "Capture",
    clip_to_capture: bool,
) -> TargetResolution:
    if not clip_to_capture:
        return target
    return clip_resolution_to_capture(target, capture)


def clip_resolution_to_capture(
    target: TargetResolution,
    capture: "Capture",
) -> TargetResolution:
    clipped = _clip_rect_to_capture(target.rect, capture)
    if clipped is None:
        return replace(target, rejected_reason="target outside capture")
    if clipped == target.rect:
        return target
    return replace(target, rect=clipped)


def _guard_revalidated_target(
    *,
    decision: "LiveHelpDecision",
    capture: "Capture",
    candidates: list[ControlCandidate],
    previous_target: TargetResolution,
    previous_capture: Any = None,
    previous_candidates: list[ControlCandidate] | None = None,
    target: TargetResolution,
    snapper: Snapper,
) -> TargetResolution:
    """Reject rechecks that no longer point near the originally resolved target."""
    if target.rejected_reason:
        return target
    if _revalidated_row_identity_changed(previous_target, target, candidates):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_row_window_context_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_tabular_identity_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_tabular_context_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_tabular_window_context_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_action_identity_changed(
        decision.instruction,
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_action_context_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_action_window_context_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_action_visual_context_changed(
        previous_target,
        target,
        previous_capture,
        capture,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_control_context_changed(
        decision.instruction,
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_control_window_context_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_control_identity_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_surface_identity_changed(
        previous_target,
        target,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _contextless_generic_action_visual_context_changed(
        previous_target,
        target,
        previous_capture,
        capture,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_ephemeral_target_context_changed(
        previous_target,
        target,
        previous_capture,
        capture,
        previous_candidates or [],
        candidates,
    ):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _model_only_revalidation_lacks_fresh_evidence(previous_target, target):
        return replace(target, rejected_reason="current screen recheck target changed")
    if _revalidated_candidate_id_replaced(decision, previous_target, target):
        return replace(target, rejected_reason="current screen recheck target changed")
    if decision.target_id and target.source == "target_id":
        independent = resolve_help_target(
            replace(decision, target_id=""),
            capture,
            candidates,
            snapper=snapper,
            clip_to_capture=False,
        )
        if independent.rejected_reason or not _same_revalidated_target(target, independent):
            return replace(target, rejected_reason="current screen recheck target changed")
    if _same_revalidation_geometry(previous_target.rect, target.rect):
        return target
    if (
        _within_revalidation_drift(previous_target.rect, target.rect)
        and _revalidation_overlap_fraction(previous_target.rect, target.rect)
        >= MIN_REVALIDATION_OVERLAP_FRACTION
    ):
        return target
    if not decision.target_id or target.source != "target_id":
        return replace(target, rejected_reason="current screen recheck target changed")

    if _same_revalidation_geometry(previous_target.rect, target.rect):
        return target
    return replace(target, rejected_reason="current screen recheck target changed")


def _revalidated_candidate_id_replaced(
    decision: "LiveHelpDecision",
    previous_target: TargetResolution,
    target: TargetResolution,
) -> bool:
    if not decision.target_id:
        return False
    if not previous_target.target_id or not target.target_id:
        return False
    if previous_target.target_id == target.target_id:
        return False
    if previous_target.source not in UIA_BACKED_TARGET_SOURCES:
        return False
    if target.source not in UIA_BACKED_TARGET_SOURCES:
        return False
    return target.source != "target_id"


def _revalidated_row_identity_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id or previous_target.target_id != target.target_id:
        return False
    current = next((candidate for candidate in candidates if candidate.id == target.target_id), None)
    if current is None or current.control_type not in ROW_REVALIDATION_CONTROL_TYPES:
        return False
    previous_tokens = _row_identity_tokens(previous_target.matched_text)
    current_tokens = _row_identity_tokens(target.matched_text or current.text)
    if not previous_tokens or not current_tokens:
        return False
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity <= 0.5


def _revalidated_row_window_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in ROW_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in ROW_REVALIDATION_CONTROL_TYPES:
        return False
    return _revalidation_window_context_changed(previous, current)


def _model_only_revalidation_lacks_fresh_evidence(
    previous_target: TargetResolution,
    target: TargetResolution,
) -> bool:
    return previous_target.source == "model" and target.source == "model"


def _row_identity_tokens(text: str) -> set[str]:
    return _tokenize_control(text or "") - ROW_REVALIDATION_GENERIC_WORDS


def _revalidated_tabular_identity_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in TABULAR_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in TABULAR_REVALIDATION_CONTROL_TYPES:
        return False
    previous_tokens = _tabular_identity_tokens(previous_target.matched_text or previous.text)
    current_tokens = _tabular_identity_tokens(target.matched_text or current.text)
    if not previous_tokens or not current_tokens:
        return False
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity < 0.5


def _revalidated_tabular_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in TABULAR_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in TABULAR_REVALIDATION_CONTROL_TYPES:
        return False
    previous_tokens = _control_tabular_context_revalidation_tokens(previous, previous_candidates)
    current_tokens = _control_tabular_context_revalidation_tokens(current, candidates)
    if not previous_tokens:
        return bool(current_tokens)
    if not current_tokens:
        return True
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity <= 0.75


def _revalidated_tabular_window_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in TABULAR_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in TABULAR_REVALIDATION_CONTROL_TYPES:
        return False
    return _revalidation_window_context_changed(previous, current)


def _tabular_identity_tokens(text: str) -> set[str]:
    return (
        _tokenize_control(text or "") | _tokens_from_text(text or "")
    ) - TABULAR_REVALIDATION_GENERIC_WORDS


def _revalidated_action_identity_changed(
    instruction: str,
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if current.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if previous.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    previous_tokens = _action_identity_revalidation_tokens(
        previous,
        previous_target.matched_text,
    )
    current_tokens = _action_identity_revalidation_tokens(
        current,
        target.matched_text,
    )
    if not previous_tokens:
        return False
    if not current_tokens:
        return True
    overlap = previous_tokens & current_tokens
    if _revalidated_action_label_gained_unrequested_tokens(
        instruction,
        previous_tokens,
        current_tokens,
    ):
        return True
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity < 0.5


def _revalidated_action_label_gained_unrequested_tokens(
    instruction: str,
    previous_tokens: set[str],
    current_tokens: set[str],
) -> bool:
    added_tokens = current_tokens - previous_tokens
    if not added_tokens:
        return False
    requested_tokens = (
        _tokens_from_text(instruction or "")
        | _tokenize_instruction(instruction or "")
    ) - ACTION_IDENTITY_REVALIDATION_GENERIC_WORDS
    return not added_tokens <= requested_tokens


def _revalidation_candidate_for_target(
    target: TargetResolution,
    candidates: list[ControlCandidate],
) -> ControlCandidate | None:
    if target.target_id:
        candidate = next((item for item in candidates if item.id == target.target_id), None)
        if candidate is not None:
            return candidate
    geometry_matches = [
        candidate
        for candidate in candidates
        if _rect_iou(candidate.rect, target.rect) >= 0.80
    ]
    if len(geometry_matches) == 1:
        return geometry_matches[0]
    return None


def _action_identity_revalidation_tokens(
    candidate: ControlCandidate | None,
    matched_text: str,
) -> set[str]:
    parts = [matched_text or ""]
    if candidate is not None:
        parts.extend([candidate.text or "", candidate.automation_id or ""])
    text = " ".join(part for part in parts if part)
    if not text:
        return set()
    return (
        _tokenize_control(text)
        | _tokens_from_text(text)
    ) - ACTION_IDENTITY_REVALIDATION_GENERIC_WORDS


def _revalidated_control_identity_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    previous_type = previous.control_type.lower()
    current_type = current.control_type.lower()
    if current_type not in CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES:
        return False
    if previous_type not in CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES:
        return False
    if previous_type != current_type:
        return True
    previous_tokens = _control_identity_revalidation_tokens(
        previous,
        previous_target.matched_text,
    )
    current_tokens = _control_identity_revalidation_tokens(
        current,
        target.matched_text,
    )
    if not previous_tokens:
        return False
    if not current_tokens:
        return True
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity < 0.5


def _revalidated_control_context_changed(
    instruction: str,
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    previous_type = previous.control_type.lower()
    current_type = current.control_type.lower()
    if previous_type not in CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES:
        return False
    if current_type not in CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES:
        return False
    if previous_type != current_type:
        return False
    previous_tokens = _control_context_revalidation_tokens(previous, previous_candidates)
    current_tokens = _control_context_revalidation_tokens(current, candidates)
    if not previous_tokens:
        return bool(current_tokens)
    if not current_tokens:
        return True
    overlap = previous_tokens & current_tokens
    previous_only = previous_tokens - current_tokens
    requested_tokens = (
        _tokens_from_text(instruction or "")
        | _tokenize_instruction(instruction or "")
    ) - CONTROL_CONTEXT_REVALIDATION_GENERIC_WORDS
    if previous_only & requested_tokens:
        return True
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity <= 0.75


def _revalidated_control_window_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    previous_type = previous.control_type.lower()
    current_type = current.control_type.lower()
    if previous_type not in CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES:
        return False
    if current_type not in CONTROL_IDENTITY_REVALIDATION_CONTROL_TYPES:
        return False
    if previous_type != current_type:
        return False
    return _revalidation_window_context_changed(previous, current)


def _control_identity_revalidation_tokens(
    candidate: ControlCandidate | None,
    matched_text: str,
) -> set[str]:
    parts = [matched_text or ""]
    if candidate is not None:
        parts.extend([candidate.text or "", candidate.automation_id or ""])
    text = " ".join(part for part in parts if part)
    if not text:
        return set()
    return (
        _tokenize_control(text)
        | _tokens_from_text(text)
    ) - CONTROL_IDENTITY_REVALIDATION_GENERIC_WORDS


def _control_context_revalidation_tokens(
    target: ControlCandidate,
    candidates: list[ControlCandidate],
) -> set[str]:
    best: tuple[float, ControlCandidate] | None = None
    tokens: set[str] = set()
    for candidate in candidates:
        if candidate.id == target.id:
            continue
        if candidate.control_type.lower() not in CONTROL_CONTEXT_LABEL_TYPES:
            continue
        if not _same_revalidation_context_window(target, candidate):
            continue
        tokens = _control_label_context_tokens(candidate)
        if not tokens:
            continue
        score = _control_context_label_score(target.rect, candidate.rect)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is not None:
        tokens.update(_control_label_context_tokens(best[1]))
    tokens.update(_control_section_context_revalidation_tokens(target, candidates))
    tokens.update(_control_tabular_context_revalidation_tokens(target, candidates))
    return tokens


def _control_section_context_revalidation_tokens(
    target: ControlCandidate,
    candidates: list[ControlCandidate],
) -> set[str]:
    target_area = max(1, target.rect[2] * target.rect[3])
    best_area: int | None = None
    best_tokens: set[str] = set()
    for candidate in candidates:
        if candidate.id == target.id:
            continue
        if candidate.control_type.lower() not in CONTROL_CONTEXT_SECTION_TYPES:
            continue
        if not _same_revalidation_context_window(target, candidate):
            continue
        if not _rect_contains(_expand_rect(candidate.rect, 4), target.rect):
            continue
        area = max(1, candidate.rect[2] * candidate.rect[3])
        if area <= target_area:
            continue
        tokens = _control_label_context_tokens(candidate)
        if not tokens:
            continue
        if best_area is None or area < best_area:
            best_area = area
            best_tokens = tokens
    return best_tokens


def _control_tabular_context_revalidation_tokens(
    target: ControlCandidate,
    candidates: list[ControlCandidate],
) -> set[str]:
    tokens: set[str] = set()
    row = _control_containing_row_context(target, candidates)
    if row is not None:
        tokens.update(_control_label_context_tokens(row))
    header = _control_aligned_header_context(target, candidates)
    if header is not None:
        tokens.update(_control_label_context_tokens(header))
    return tokens


def _control_containing_row_context(
    target: ControlCandidate,
    candidates: list[ControlCandidate],
) -> ControlCandidate | None:
    target_area = max(1, target.rect[2] * target.rect[3])
    best: tuple[int, ControlCandidate] | None = None
    for candidate in candidates:
        if candidate.id == target.id:
            continue
        if candidate.control_type.lower() not in CONTROL_CONTEXT_ROW_TYPES:
            continue
        if not _same_revalidation_context_window(target, candidate):
            continue
        if not _rect_contains(_expand_rect(candidate.rect, 3), target.rect):
            continue
        area = max(1, candidate.rect[2] * candidate.rect[3])
        if area <= target_area:
            continue
        if best is None or area < best[0]:
            best = (area, candidate)
    return best[1] if best is not None else None


def _control_aligned_header_context(
    target: ControlCandidate,
    candidates: list[ControlCandidate],
) -> ControlCandidate | None:
    target_x, target_y, target_width, target_height = target.rect
    if min(target_width, target_height) <= 0:
        return None
    target_right = target_x + target_width
    best: tuple[float, ControlCandidate] | None = None
    for candidate in candidates:
        if candidate.id == target.id:
            continue
        if candidate.control_type.lower() != "headeritem":
            continue
        if not _same_revalidation_context_window(target, candidate):
            continue
        header_x, header_y, header_width, header_height = candidate.rect
        if min(header_width, header_height) <= 0:
            continue
        if header_y >= target_y:
            continue
        header_right = header_x + header_width
        horizontal_overlap = min(target_right, header_right) - max(target_x, header_x)
        if horizontal_overlap <= 0:
            continue
        overlap_fraction = horizontal_overlap / max(1, min(target_width, header_width))
        if overlap_fraction < 0.45:
            continue
        vertical_gap = target_y - (header_y + header_height)
        if vertical_gap > max(96, target_height * 4):
            continue
        score = overlap_fraction - min(0.4, vertical_gap / max(1.0, target_height * 8.0))
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _same_revalidation_context_window(
    target: ControlCandidate,
    context: ControlCandidate,
) -> bool:
    if target.window_rank != context.window_rank:
        return False
    return not _window_title_identity_changed(target.window_title, context.window_title)


def _control_label_context_tokens(candidate: ControlCandidate) -> set[str]:
    text = " ".join(
        part
        for part in (candidate.text or "", candidate.automation_id or "")
        if part
    )
    if not text:
        return set()
    return (
        _tokens_from_text(text)
        | _tokenize_control(text)
    ) - CONTROL_CONTEXT_REVALIDATION_GENERIC_WORDS


def _control_context_label_score(
    control_rect: tuple[int, int, int, int],
    label_rect: tuple[int, int, int, int],
) -> float:
    control_x, control_y, control_width, control_height = control_rect
    label_x, label_y, label_width, label_height = label_rect
    if min(control_width, control_height, label_width, label_height) <= 0:
        return 0.0
    control_right = control_x + control_width
    control_bottom = control_y + control_height
    label_right = label_x + label_width
    label_bottom = label_y + label_height
    control_center_y = control_y + control_height / 2
    label_center_y = label_y + label_height / 2
    vertical_overlap = min(control_bottom, label_bottom) - max(control_y, label_y)
    if vertical_overlap >= min(control_height, label_height) * 0.45:
        horizontal_gap = max(control_x - label_right, label_x - control_right, 0)
        if horizontal_gap <= max(48, min(220, max(control_height, label_height) * 6)):
            y_penalty = abs(control_center_y - label_center_y) / max(1.0, control_height)
            return 2.0 - min(1.0, y_penalty)
    if label_bottom <= control_y:
        vertical_gap = control_y - label_bottom
        horizontal_overlap = min(control_right, label_right) - max(control_x, label_x)
        left_aligned = abs(label_x - control_x) <= max(16, min(control_width, label_width) * 0.30)
        center_aligned = abs(
            (label_x + label_right) / 2 - (control_x + control_right) / 2
        ) <= max(32, min(control_width, label_width) * 0.45)
        if vertical_gap <= max(36, control_height * 1.2) and (
            horizontal_overlap > 0 or left_aligned or center_aligned
        ):
            return 1.0 - min(0.9, vertical_gap / max(1.0, control_height * 2))
    return 0.0


def _revalidated_action_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if current.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if previous.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    previous_tokens = _meaningful_action_context_revalidation_tokens(
        _action_context_revalidation_tokens(previous, previous_candidates)
    )
    current_tokens = _meaningful_action_context_revalidation_tokens(
        _action_context_revalidation_tokens(current, candidates)
    )
    if not previous_tokens and not current_tokens:
        return False
    if not previous_tokens or not current_tokens:
        return True
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity < 0.5


def _revalidated_action_window_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    return _revalidation_window_context_changed(previous, current)


def _revalidated_action_visual_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_capture: Any,
    capture: "Capture",
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if previous_capture is None:
        return False
    if not previous_target.target_id or previous_target.target_id != target.target_id:
        return False
    if not _ephemeral_revalidation_target_id(target.target_id):
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if previous.control_type != current.control_type:
        return True
    if not _same_revalidation_geometry(previous_target.rect, target.rect):
        return False
    previous_context = _meaningful_action_context_revalidation_tokens(
        _action_context_revalidation_tokens(previous, previous_candidates)
    )
    current_context = _meaningful_action_context_revalidation_tokens(
        _action_context_revalidation_tokens(current, candidates)
    )
    if previous_context or current_context:
        if not previous_context or not current_context:
            return True
        overlap = previous_context & current_context
        similarity = len(overlap) / max(1, max(len(previous_context), len(current_context)))
        return similarity <= 0.5
    return _revalidation_visual_context_changed(
        previous_target.rect,
        target.rect,
        previous_capture,
        capture,
        diff_floor=EPHEMERAL_ACTION_REVALIDATION_CONTEXT_DIFF_FLOOR,
    )


def _revalidated_surface_identity_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    previous_type = previous.control_type.lower()
    current_type = current.control_type.lower()
    if previous_type not in SURFACE_REVALIDATION_CONTROL_TYPES:
        return False
    if current_type not in SURFACE_REVALIDATION_CONTROL_TYPES:
        return False
    if previous_type != current_type:
        return True
    previous_tokens = _surface_identity_revalidation_tokens(previous, previous_target.matched_text)
    current_tokens = _surface_identity_revalidation_tokens(current, target.matched_text)
    if not previous_tokens:
        return False
    if not current_tokens:
        return True
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity <= 0.5


def _surface_identity_revalidation_tokens(
    candidate: ControlCandidate,
    matched_text: str,
) -> set[str]:
    text = " ".join(
        part
        for part in (matched_text or "", candidate.text or "", candidate.automation_id or "")
        if part
    )
    if not text:
        return set()
    return (
        _tokenize_control(text)
        | _tokens_from_text(text)
    ) - SURFACE_IDENTITY_REVALIDATION_GENERIC_WORDS


def _revalidation_window_context_changed(
    previous: ControlCandidate,
    current: ControlCandidate,
) -> bool:
    if previous.window_rank != current.window_rank:
        return True
    return _window_title_identity_changed(previous.window_title, current.window_title)


def _revalidated_ephemeral_target_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_capture: Any,
    capture: "Capture",
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if not previous_target.target_id or previous_target.target_id != target.target_id:
        return False
    if not _ephemeral_revalidation_target_id(target.target_id):
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type.lower() != current.control_type.lower():
        return True
    if _strict_window_identity_missing_or_changed(previous, current):
        return True
    if previous_capture is None:
        return False
    if not _same_revalidation_geometry(previous_target.rect, target.rect):
        return False
    return _revalidation_visual_context_changed(previous_target.rect, target.rect, previous_capture, capture)


def _ephemeral_revalidation_target_id(target_id: str) -> bool:
    return bool(re.fullmatch(r"c\d+", (target_id or "").strip()))


def _strict_window_identity_missing_or_changed(
    previous: ControlCandidate,
    current: ControlCandidate,
) -> bool:
    if previous.window_rank != current.window_rank:
        return True
    previous_title = (previous.window_title or "").strip()
    current_title = (current.window_title or "").strip()
    if bool(previous_title) != bool(current_title):
        return True
    if previous_title and current_title:
        return _window_title_identity_changed(previous_title, current_title)
    return False


def _revalidation_visual_context_changed(
    previous_rect: tuple[int, int, int, int],
    current_rect: tuple[int, int, int, int],
    previous_capture: Any,
    capture: "Capture",
    *,
    diff_floor: float = GENERIC_ACTION_REVALIDATION_CONTEXT_DIFF_FLOOR,
) -> bool:
    previous_crop = _capture_rect_image(
        previous_capture,
        _expand_rect(previous_rect, GENERIC_ACTION_REVALIDATION_CONTEXT_MARGIN_PX),
    )
    current_crop = _capture_rect_image(
        capture,
        _expand_rect(current_rect, GENERIC_ACTION_REVALIDATION_CONTEXT_MARGIN_PX),
    )
    if previous_crop is None or current_crop is None:
        return False
    if previous_crop.size != current_crop.size:
        current_crop = current_crop.resize(previous_crop.size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(previous_crop.convert("RGB"), current_crop.convert("RGB"))
    stat = ImageStat.Stat(diff)
    normalized = sum(stat.mean) / max(1, len(stat.mean) * 255)
    return normalized > diff_floor


def _window_title_identity_changed(previous_title: str, current_title: str) -> bool:
    previous = (previous_title or "").strip()
    current = (current_title or "").strip()
    if not previous or not current:
        return False
    if previous.casefold() == current.casefold():
        return False
    previous_tokens = _window_title_identity_tokens(previous)
    current_tokens = _window_title_identity_tokens(current)
    if not previous_tokens or not current_tokens:
        return False
    overlap = previous_tokens & current_tokens
    similarity = len(overlap) / max(1, max(len(previous_tokens), len(current_tokens)))
    return similarity < 0.5


def _window_title_identity_tokens(title: str) -> set[str]:
    parts = [part.strip() for part in re.split(r"\s+[-|:]\s+", title) if part.strip()]
    if len(parts) > 1:
        title = parts[0]
    return (
        _tokens_from_text(title)
        | _tokenize_control(title)
    ) - WINDOW_TITLE_REVALIDATION_GENERIC_WORDS


def _contextless_generic_action_visual_context_changed(
    previous_target: TargetResolution,
    target: TargetResolution,
    previous_capture: Any,
    capture: "Capture",
    previous_candidates: list[ControlCandidate],
    candidates: list[ControlCandidate],
) -> bool:
    if previous_capture is None:
        return False
    if not previous_target.target_id and not target.target_id:
        return False
    previous = _revalidation_candidate_for_target(previous_target, previous_candidates)
    current = _revalidation_candidate_for_target(target, candidates)
    if previous is None or current is None:
        return False
    if previous.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if current.control_type not in ACTION_REVALIDATION_CONTROL_TYPES:
        return False
    if _action_specific_revalidation_tokens(previous, previous_target.matched_text):
        return False
    if _action_specific_revalidation_tokens(current, target.matched_text):
        return False
    if _meaningful_action_context_revalidation_tokens(
        _action_context_revalidation_tokens(previous, previous_candidates)
    ):
        return False
    if _meaningful_action_context_revalidation_tokens(
        _action_context_revalidation_tokens(current, candidates)
    ):
        return False
    previous_crop = _capture_rect_image(
        previous_capture,
        _expand_rect(previous_target.rect, GENERIC_ACTION_REVALIDATION_CONTEXT_MARGIN_PX),
    )
    current_crop = _capture_rect_image(
        capture,
        _expand_rect(target.rect, GENERIC_ACTION_REVALIDATION_CONTEXT_MARGIN_PX),
    )
    if previous_crop is None or current_crop is None:
        return False
    if previous_crop.size != current_crop.size:
        current_crop = current_crop.resize(previous_crop.size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(previous_crop.convert("RGB"), current_crop.convert("RGB"))
    stat = ImageStat.Stat(diff)
    normalized = sum(stat.mean) / max(1, len(stat.mean) * 255)
    return normalized > GENERIC_ACTION_REVALIDATION_CONTEXT_DIFF_FLOOR


def _meaningful_action_context_revalidation_tokens(tokens: set[str]) -> set[str]:
    return set(tokens) - WEAK_ACTION_CONTEXT_REVALIDATION_WORDS


def _action_specific_revalidation_tokens(
    candidate: ControlCandidate,
    matched_text: str,
) -> set[str]:
    return _action_identity_revalidation_tokens(candidate, matched_text) - GENERIC_ACTION_REVALIDATION_WORDS


def _capture_rect_image(
    capture: "Capture",
    rect: tuple[int, int, int, int],
) -> Image.Image | None:
    image_rect = _screen_rect_to_image_rect(capture, rect)
    clipped = _intersection_rect(image_rect, (0, 0, capture.width, capture.height))
    if clipped is None:
        return None
    try:
        image = Image.open(io.BytesIO(capture.png_bytes))
        image.load()
    except Exception:
        return None
    x, y, width, height = clipped
    return image.convert("RGB").crop((x, y, x + width, y + height))


def _screen_rect_to_image_rect(
    capture: "Capture",
    rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    left = int((x - capture.monitor_left) * capture.scale)
    top = int((y - capture.monitor_top) * capture.scale)
    scaled_width = max(1, int(width * capture.scale))
    scaled_height = max(1, int(height * capture.scale))
    return (left, top, scaled_width, scaled_height)


def _action_context_revalidation_tokens(
    target: ControlCandidate,
    candidates: list[ControlCandidate],
) -> set[str]:
    target_tokens = _surface_evidence_tokens(target)
    contexts: list[ControlCandidate] = []
    for candidate in candidates:
        if candidate.id == target.id:
            continue
        if candidate.control_type not in ACTION_CONTEXT_REVALIDATION_CONTROL_TYPES:
            continue
        if not _same_revalidation_context_window(target, candidate):
            continue
        if not _rect_contains(_expand_rect(candidate.rect, 4), target.rect):
            continue
        contexts.append(candidate)
    contexts.sort(key=lambda candidate: candidate.rect[2] * candidate.rect[3])
    for context in contexts:
        tokens = _surface_evidence_tokens(context) - target_tokens
        tokens |= _contained_action_context_text_tokens(
            target,
            context,
            candidates,
            target_tokens,
        )
        tokens -= ACTION_CONTEXT_REVALIDATION_GENERIC_WORDS
        if tokens:
            return tokens
    return set()


def _contained_action_context_text_tokens(
    target: ControlCandidate,
    context: ControlCandidate,
    candidates: list[ControlCandidate],
    target_tokens: set[str],
) -> set[str]:
    tokens: set[str] = set()
    for candidate in candidates:
        if candidate.id == target.id:
            continue
        if candidate.control_type not in ACTION_CONTEXT_TEXT_CONTROL_TYPES:
            continue
        if not _same_revalidation_context_window(target, candidate):
            continue
        if not _rect_contains(_expand_rect(context.rect, 4), candidate.rect):
            continue
        if _rect_iou(candidate.rect, target.rect) >= 0.50:
            continue
        tokens |= _surface_evidence_tokens(candidate)
    return tokens - target_tokens - ACTION_CONTEXT_REVALIDATION_GENERIC_WORDS


def _same_revalidated_target(a: TargetResolution, b: TargetResolution) -> bool:
    if _same_revalidation_geometry(a.rect, b.rect):
        return True
    if a.target_id and b.target_id and a.target_id == b.target_id:
        return True
    return False


def _same_revalidation_geometry(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    if _rect_iou(a, b) >= 0.80:
        return True
    if _intersection_rect(a, b) is None:
        return False
    ax, ay = _rect_center(a)
    bx, by = _rect_center(b)
    max_drift = max(8.0, min(a[2], a[3], b[2], b[3]) * 0.35)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return distance <= max_drift


def _revalidation_overlap_fraction(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    inter = _intersection_rect(a, b)
    if inter is None:
        return 0.0
    inter_area = inter[2] * inter[3]
    a_area = max(1, a[2] * a[3])
    b_area = max(1, b[2] * b[3])
    return inter_area / max(1, min(a_area, b_area))


def _within_revalidation_drift(
    previous: tuple[int, int, int, int],
    current: tuple[int, int, int, int],
) -> bool:
    px, py = _rect_center(previous)
    cx, cy = _rect_center(current)
    max_edge = max(previous[2], previous[3], current[2], current[3], 1)
    max_drift = max(32.0, max_edge * 0.75)
    distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    return distance <= max_drift


def _rect_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    inter = _intersection_rect(a, b)
    if inter is None:
        return 0.0
    inter_area = inter[2] * inter[3]
    a_area = max(1, a[2] * a[3])
    b_area = max(1, b[2] * b[3])
    return inter_area / max(1, a_area + b_area - inter_area)


def _rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = rect
    return (x + width / 2.0, y + height / 2.0)


def _foreground_candidate_covering_reason(
    target: TargetResolution,
    candidates: list[ControlCandidate],
    previous_candidates: list[ControlCandidate] | None = None,
) -> str:
    selected = _revalidation_candidate_for_target(target, candidates)
    if selected is None:
        return ""
    selected_rank = selected.window_rank
    for candidate in candidates:
        if candidate.id == selected.id:
            continue
        if candidate.window_rank > selected_rank:
            continue
        if (
            candidate.window_rank == selected_rank
            and candidate.control_type == selected.control_type
            and _rect_iou(candidate.rect, selected.rect) >= 0.98
            and _same_rank_exact_duplicate_was_previously_present(
                candidate,
                previous_candidates or [],
            )
        ):
            continue
        if candidate.window_rank == selected_rank and not _same_rank_candidate_can_cover_selected(
            candidate,
            selected,
            previous_candidates or [],
        ):
            continue
        overlap = _target_coverage_fraction(candidate.rect, target.rect)
        if overlap >= FOREGROUND_COVERAGE_MIN_FRACTION:
            return "target covered before overlay"
        if (
            overlap >= FOREGROUND_CENTER_COVERAGE_MIN_FRACTION
            and _rect_contains_point(candidate.rect, _rect_center(target.rect))
        ):
            return "target covered before overlay"
    return ""


def _same_rank_candidate_can_cover_selected(
    candidate: ControlCandidate,
    selected: ControlCandidate,
    previous_candidates: list[ControlCandidate],
) -> bool:
    if candidate.control_type in SAME_RANK_COVERING_CONTROL_TYPES:
        if (
            _same_rank_surface_was_previously_present(candidate, previous_candidates)
            and _same_rank_previous_owner_contains_selected(candidate, selected)
        ):
            return False
        return True
    if candidate.control_type in SAME_RANK_OWNING_COVERING_CONTROL_TYPES:
        if not _same_rank_surface_was_previously_present(candidate, previous_candidates):
            return True
        if (
            _same_rank_surface_was_previously_present(candidate, previous_candidates)
            and _same_rank_previous_owner_contains_selected(candidate, selected)
        ):
            return False
        return not _same_rank_surface_owns_selected(candidate, selected)
    if candidate.control_type not in SAME_RANK_TRANSIENT_SURFACE_CONTROL_TYPES:
        return False
    if _same_rank_surface_was_previously_present(
        candidate,
        previous_candidates,
    ) and _same_rank_surface_owns_selected(candidate, selected):
        return False
    if (
        _same_rank_surface_was_previously_present(candidate, previous_candidates)
        and _same_rank_previous_owner_contains_selected(candidate, selected)
    ):
        return False
    if not _same_rank_surface_was_previously_present(candidate, previous_candidates):
        return True
    tokens = (
        _tokenize_control(candidate.text)
        | _tokenize_control(candidate.automation_id)
        | _tokenize_control(candidate.window_title)
    )
    return bool(tokens & SAME_RANK_TRANSIENT_SURFACE_WORDS)


def _same_rank_exact_duplicate_was_previously_present(
    candidate: ControlCandidate,
    previous_candidates: list[ControlCandidate],
) -> bool:
    for previous in previous_candidates:
        if previous.window_rank != candidate.window_rank:
            continue
        if previous.control_type != candidate.control_type:
            continue
        if _rect_iou(previous.rect, candidate.rect) < 0.98:
            continue
        if previous.text != candidate.text:
            continue
        if previous.automation_id != candidate.automation_id:
            continue
        if previous.window_title != candidate.window_title:
            continue
        return True
    return False


def _same_rank_previous_owner_contains_selected(
    candidate: ControlCandidate,
    selected: ControlCandidate,
) -> bool:
    expanded = _expand_rect(candidate.rect, 4)
    if not (
        _rect_contains(expanded, selected.rect)
        or _rect_contains_point(expanded, _rect_center(selected.rect))
    ):
        return False
    selected_window = (selected.window_title or "").strip().lower()
    candidate_surface_names = {
        (candidate.text or "").strip().lower(),
        (candidate.automation_id or "").strip().lower(),
        (candidate.window_title or "").strip().lower(),
    }
    if (
        selected_window
        and candidate.control_type in SAME_RANK_TRANSIENT_SURFACE_CONTROL_TYPES
        and selected_window not in candidate_surface_names
    ):
        return False
    candidate_area = max(1, candidate.rect[2] * candidate.rect[3])
    selected_area = max(1, selected.rect[2] * selected.rect[3])
    if candidate_area <= selected_area:
        return False
    if candidate.depth < selected.depth:
        return True
    candidate_tokens = (
        _tokenize_control(candidate.text)
        | _tokenize_control(candidate.automation_id)
        | _tokenize_control(candidate.window_title)
    )
    selected_tokens = (
        _tokenize_control(selected.text)
        | _tokenize_control(selected.automation_id)
        | _tokenize_control(selected.window_title)
    )
    if candidate_tokens and selected_tokens and candidate_tokens & selected_tokens:
        return True
    return candidate_area / selected_area >= SAME_RANK_PREVIOUS_OWNER_MIN_AREA_RATIO


def _same_rank_surface_owns_selected(
    candidate: ControlCandidate,
    selected: ControlCandidate,
) -> bool:
    if not _rect_contains(_expand_rect(candidate.rect, 4), selected.rect):
        return False
    if candidate.control_type == "menu" and selected.control_type == "menuitem":
        return True
    selected_window = (selected.window_title or "").strip().lower()
    surface_names = {
        (candidate.text or "").strip().lower(),
        (candidate.automation_id or "").strip().lower(),
        (candidate.window_title or "").strip().lower(),
    }
    if selected_window and selected_window in surface_names:
        return True
    candidate_window = (candidate.window_title or "").strip().lower()
    if (
        candidate_window
        and selected_window
        and candidate_window == selected_window
        and candidate.depth < selected.depth
    ):
        return True
    surface_tokens = (
        _tokenize_control(candidate.text)
        | _tokenize_control(candidate.automation_id)
        | _tokenize_control(candidate.window_title)
    ) - SAME_RANK_TRANSIENT_SURFACE_WORDS - {"app", "application", "window"}
    selected_context_tokens = _tokenize_control(selected.window_title)
    return bool(surface_tokens and selected_context_tokens and surface_tokens & selected_context_tokens)


def _same_rank_surface_was_previously_present(
    candidate: ControlCandidate,
    previous_candidates: list[ControlCandidate],
) -> bool:
    for previous in previous_candidates:
        if previous.control_type != candidate.control_type:
            continue
        if previous.id == candidate.id:
            return True
        if _rect_iou(previous.rect, candidate.rect) >= 0.80 and (
            previous.text == candidate.text
            or previous.automation_id == candidate.automation_id
        ):
            return True
    return False


def _target_coverage_fraction(
    covering_rect: tuple[int, int, int, int],
    target_rect: tuple[int, int, int, int],
) -> float:
    inter = _intersection_rect(covering_rect, target_rect)
    if inter is None:
        return 0.0
    target_area = max(1, target_rect[2] * target_rect[3])
    return (inter[2] * inter[3]) / target_area


def _rect_contains_point(
    rect: tuple[int, int, int, int],
    point: tuple[float, float],
) -> bool:
    x, y, width, height = rect
    px, py = point
    return x <= px <= x + width and y <= py <= y + height


def _clip_rect_to_capture(
    rect: tuple[int, int, int, int],
    capture: "Capture",
) -> tuple[int, int, int, int] | None:
    scale = max(capture.scale, 0.001)
    bounds = (
        capture.monitor_left,
        capture.monitor_top,
        int(capture.width / scale),
        int(capture.height / scale),
    )
    return _intersection_rect(rect, bounds)


def _intersection_rect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2 - ix1, iy2 - iy1)


@dataclass
class _HelpRun:
    """Per-thread state so a replacement session cannot steal old events."""

    cancelled: Event = field(default_factory=Event)
    click_inside: Event = field(default_factory=Event)
    check_now: Event = field(default_factory=Event)
    thread: Thread | None = None


class HelpSession(QObject):
    ghost_clear = pyqtSignal()
    highlight_show = pyqtSignal(int, int, int, int, str)
    highlight_clear = pyqtSignal()
    chat_message = pyqtSignal(str)
    chat_status = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    step_skipped = pyqtSignal(str)
    target_diagnostic = pyqtSignal(dict)

    def __init__(
        self,
        agent: "HelplerAgent",
        controller: "ComputerController",
        *,
        parent: QObject | None = None,
        capture_provider: CaptureProvider = capture_active_monitor,
        candidate_provider: CandidateProvider = collect_control_candidates,
        snapper: Snapper = snap_to_control,
        overlay_clear_barrier: OverlayClearBarrier | None = None,
        ocr_text_provider: OcrTextProvider | None = None,
        enable_ocr_text_verification: bool = True,
    ) -> None:
        super().__init__(parent)
        self._agent = agent
        self._controller = controller
        self._capture_provider = capture_provider
        self._candidate_provider = candidate_provider
        self._snapper = snapper
        self._overlay_clear_barrier = overlay_clear_barrier
        self._ocr_text_provider = (
            ocr_text_provider
            if ocr_text_provider is not None
            else default_ocr_text_provider() if enable_ocr_text_verification else None
        )
        self._thread: Thread | None = None
        self._cancelled = Event()
        self._click_inside_event = Event()
        self._check_now_event = Event()
        self._run_lock = threading.RLock()
        self._current_run: _HelpRun | None = None
        self._run_local = threading.local()
        self._active_rect: tuple[int, int, int, int] | None = None
        self._rect_lock = threading.Lock()

    def cancel(self) -> None:
        with self._run_lock:
            run = self._current_run
            thread = run.thread if run is not None else self._thread
        if run is not None:
            run.cancelled.set()
            run.click_inside.set()
            run.check_now.set()
        else:
            self._cancelled.set()
            self._click_inside_event.set()
            self._check_now_event.set()
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.1)

    def notify_user_click(self, screen_x: int, screen_y: int) -> None:
        """Called from the global mouse listener whenever the user clicks.

        Any click forces an immediate re-evaluation. A click inside the active
        rect (expanded by CLICK_HIT_MARGIN_PX) is recorded so the next outcome
        note can tell the model "the user followed the suggestion"; clicks
        outside are recorded as "the user clicked elsewhere".
        """
        with self._rect_lock:
            rect = self._active_rect
        with self._run_lock:
            run = self._current_run
        click_inside_event = run.click_inside if run is not None else self._click_inside_event
        check_now_event = run.check_now if run is not None else self._check_now_event
        if rect is None:
            check_now_event.set()
            return
        rx, ry, rw, rh = rect
        margin = click_hit_margin(rect)
        inside = (
            (rx - margin) <= screen_x < (rx + rw + margin)
            and (ry - margin) <= screen_y < (ry + rh + margin)
        )
        if inside:
            click_inside_event.set()
        else:
            check_now_event.set()

    def start(self, message: str) -> None:
        self.cancel()
        run = _HelpRun()
        self._cancelled = run.cancelled
        self._click_inside_event = run.click_inside
        self._check_now_event = run.check_now
        thread = Thread(target=self._run, args=(message, run), daemon=True)
        run.thread = thread
        with self._run_lock:
            self._current_run = run
            self._thread = thread
        thread.start()

    def _run(self, message: str, run: _HelpRun | None = None) -> None:
        if run is None:
            with self._run_lock:
                run = self._current_run
            if run is None:
                run = _HelpRun(thread=threading.current_thread())
        self._run_local.context = run
        try:
            self._run_walkthrough(message)
        except Exception as exc:
            if not self._is_current_run(run):
                return
            log.exception("Help session crashed")
            self.failed.emit(f"Helper walkthrough failed: {exc}")
        finally:
            with self._run_lock:
                if self._current_run is run:
                    self._current_run = None
                    self._thread = None
            try:
                del self._run_local.context
            except AttributeError:
                pass

    def _run_context(self) -> _HelpRun | None:
        run = getattr(self._run_local, "context", None)
        if run is not None:
            return run
        with self._run_lock:
            return self._current_run

    def _is_current_run(self, run: _HelpRun) -> bool:
        with self._run_lock:
            return self._current_run is run

    def _run_walkthrough(self, message: str) -> None:
        if self._aborted():
            return

        history = HistoryManager()
        outcome_note = (message or "").strip() or "Help me with what's on my screen."

        for _ in range(MAX_TURNS):
            if self._aborted():
                return

            self.chat_status.emit("Looking at your screen...")
            self._clear_overlays(wait_for_flush=True)
            if self._aborted():
                return
            try:
                capture = self._capture_provider()
            except Exception as exc:
                log.exception("Screenshot failed")
                self.failed.emit(f"Couldn't capture the screen: {exc}")
                return

            candidates = self._collect_candidates(capture)
            history.add_user_turn(text=outcome_note, screenshot=capture)

            try:
                decision = self._agent.plan_next_step(
                    history,
                    control_candidates=candidates,
                    capture=capture,
                )
            except Exception as exc:
                log.exception("plan_next_step failed")
                self.failed.emit(f"Helper couldn't decide a step: {exc}")
                return

            if self._aborted():
                return

            history.add_assistant_turn(decision.history_text)

            if decision.helper_action is not None:
                self._execute_helper_action(decision.helper_action)
                if self._sleep_with_cancel(POST_ACTION_SETTLE_SEC):
                    return
                if decision.kind == "done":
                    self._end_walkthrough(decision.message or "Walkthrough complete.")
                    return
                if decision.kind == "narrate" and decision.message:
                    self.chat_message.emit(decision.message)
                outcome_note = self._outcome_after_helper_action(decision.helper_action)
                self._clear_overlays()
                continue

            if decision.kind == "done":
                self._end_walkthrough(decision.message or "Walkthrough complete.")
                return

            if decision.kind == "narrate":
                self._clear_overlays()
                message_text = decision.message or "Take a look at the screen."
                self.chat_message.emit(message_text)
                self.chat_status.emit(message_text)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_narrate(wait_outcome)
                continue

            # decision.kind == "step"
            target = resolve_help_target(
                decision,
                capture,
                candidates,
                snapper=self._snapper,
                clip_to_capture=False,
            )
            if target.rejected_reason:
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=target,
                        rejected_reason=target.rejected_reason,
                    )
                )
                log.info(
                    "Step downgraded to narrate (%s, model rect %dx%d normalized): %s",
                    target.rejected_reason,
                    decision.target_norm_width,
                    decision.target_norm_height,
                    decision.instruction,
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_downgrade(
                    decision,
                    target.rejected_reason,
                )
                continue

            try:
                capture, candidates, target = self._revalidate_target_on_current_screen(
                    decision,
                    previous_target=target,
                    previous_capture=capture,
                    previous_candidates=candidates,
                )
            except Exception:
                reason = "current screen recheck failed"
                log.exception("Pre-overlay target recheck failed")
                rejected = replace(target, rejected_reason=reason)
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=rejected,
                        rejected_reason=reason,
                    )
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(decision, reason)
                continue

            if target.rejected_reason:
                reason = f"current screen recheck: {target.rejected_reason}"
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=target,
                        rejected_reason=reason,
                    )
                )
                log.info("Step downgraded after current-screen recheck (%s): %s", reason, decision.instruction)
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(decision, reason)
                continue

            log.info(
                "Help target resolved: source=%s target_id=%s confidence=%.2f text=%r rect=%s instruction=%r",
                target.source,
                target.target_id,
                target.confidence,
                target.matched_text,
                target.rect,
                decision.instruction,
            )
            final_rect = target.rect
            target_control_type = target_control_type_for_resolution(target, candidates)
            quality = evaluate_target_quality(
                capture=capture,
                rect=final_rect,
                source=target.source,
                confidence=target.confidence,
                instruction=decision.instruction,
                target_control_type=target_control_type,
            )
            if not quality.accepted:
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=target,
                        quality=quality,
                        rejected_reason=quality.reason,
                    )
                )
                log.info(
                    "Step downgraded by target quality gate (%s, visible=%.2f activity=%.3f): %s",
                    quality.reason,
                    quality.visible_fraction,
                    quality.visual_activity,
                    decision.instruction,
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(
                    decision,
                    quality.reason,
                )
                continue
            display_target = clip_resolution_to_capture(target, capture)
            if display_target.rejected_reason:
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=display_target,
                        quality=quality,
                        rejected_reason=display_target.rejected_reason,
                    )
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(
                    decision,
                    display_target.rejected_reason,
                )
                continue
            final_rect = display_target.rect
            ocr_evidence = expected_text_evidence_for_target(display_target, candidates)
            ocr_verification = verify_target_text(
                capture=capture,
                rect=ocr_evidence.rect or final_rect,
                expected_text=ocr_evidence.text,
                control_type=target_control_type,
                provider=self._ocr_text_provider,
            )
            if not ocr_verification.accepted:
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=display_target,
                        quality=quality,
                        ocr=ocr_verification,
                        rejected_reason=ocr_verification.reason,
                    )
                )
                log.info(
                    "Step downgraded by OCR text gate (%s, expected=%r recognized=%r): %s",
                    ocr_verification.reason,
                    ocr_verification.expected_text,
                    ocr_verification.recognized_text,
                    decision.instruction,
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(
                    decision,
                    ocr_verification.reason,
                )
                continue
            try:
                final_capture, final_candidates, final_target = self._revalidate_target_on_current_screen(
                    decision,
                    previous_target=display_target,
                    previous_capture=capture,
                    previous_candidates=candidates,
                )
            except Exception:
                reason = "final pre-overlay recheck failed"
                log.exception("Final pre-overlay target recheck failed")
                rejected = replace(display_target, rejected_reason=reason)
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=capture,
                        candidates=candidates,
                        target=rejected,
                        quality=quality,
                        ocr=ocr_verification,
                        rejected_reason=reason,
                    )
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(decision, reason)
                continue
            if final_target.rejected_reason:
                reason = f"final pre-overlay recheck: {final_target.rejected_reason}"
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=final_capture,
                        candidates=final_candidates,
                        target=final_target,
                        ocr=ocr_verification,
                        rejected_reason=reason,
                    )
                )
                log.info("Step downgraded after final pre-overlay recheck (%s): %s", reason, decision.instruction)
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(decision, reason)
                continue
            coverage_reason = _foreground_candidate_covering_reason(
                final_target,
                final_candidates,
                previous_candidates=candidates,
            )
            if coverage_reason:
                rejected = replace(final_target, rejected_reason=coverage_reason)
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=final_capture,
                        candidates=final_candidates,
                        target=rejected,
                        ocr=ocr_verification,
                        rejected_reason=coverage_reason,
                    )
                )
                log.info("Step downgraded by final coverage gate (%s): %s", coverage_reason, decision.instruction)
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(decision, coverage_reason)
                continue
            final_display_target = clip_resolution_to_capture(final_target, final_capture)
            if final_display_target.rejected_reason:
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=final_capture,
                        candidates=final_candidates,
                        target=final_display_target,
                        ocr=ocr_verification,
                        rejected_reason=final_display_target.rejected_reason,
                    )
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(
                    decision,
                    final_display_target.rejected_reason,
                )
                continue
            final_rect = final_display_target.rect
            final_target_control_type = target_control_type_for_resolution(
                final_display_target,
                final_candidates,
            )
            final_quality = evaluate_target_quality(
                capture=final_capture,
                rect=final_rect,
                source=final_display_target.source,
                confidence=final_display_target.confidence,
                instruction=decision.instruction,
                target_control_type=final_target_control_type,
            )
            if not final_quality.accepted:
                self._emit_target_diagnostic(
                    build_target_diagnostic(
                        decision=decision,
                        capture=final_capture,
                        candidates=final_candidates,
                        target=final_display_target,
                        quality=final_quality,
                        ocr=ocr_verification,
                        rejected_reason=final_quality.reason,
                    )
                )
                log.info(
                    "Step downgraded by final target quality gate (%s, visible=%.2f activity=%.3f): %s",
                    final_quality.reason,
                    final_quality.visible_fraction,
                    final_quality.visual_activity,
                    decision.instruction,
                )
                self._clear_overlays()
                msg = (decision.instruction or "").strip() or "Take a look at the screen."
                self.chat_message.emit(msg)
                self.chat_status.emit(msg)
                wait_outcome = self._wait_for_progress(rect=None)
                if wait_outcome == "cancelled":
                    return
                outcome_note = self._outcome_after_quality_rejection(
                    decision,
                    final_quality.reason,
                )
                continue
            final_ocr_verification = ocr_verification
            if ocr_verification.available and ocr_verification.expected_text:
                final_ocr_evidence = expected_text_evidence_for_target(
                    final_display_target,
                    final_candidates,
                )
                final_ocr_verification = verify_target_text(
                    capture=final_capture,
                    rect=final_ocr_evidence.rect or final_rect,
                    expected_text=final_ocr_evidence.text or ocr_verification.expected_text,
                    control_type=final_target_control_type,
                    provider=self._ocr_text_provider,
                )
                if not final_ocr_verification.accepted:
                    self._emit_target_diagnostic(
                        build_target_diagnostic(
                            decision=decision,
                            capture=final_capture,
                            candidates=final_candidates,
                            target=final_display_target,
                            quality=final_quality,
                            ocr=final_ocr_verification,
                            rejected_reason=final_ocr_verification.reason,
                        )
                    )
                    log.info(
                        "Step downgraded by final OCR text gate (%s, expected=%r recognized=%r): %s",
                        final_ocr_verification.reason,
                        final_ocr_verification.expected_text,
                        final_ocr_verification.recognized_text,
                        decision.instruction,
                    )
                    self._clear_overlays()
                    msg = (decision.instruction or "").strip() or "Take a look at the screen."
                    self.chat_message.emit(msg)
                    self.chat_status.emit(msg)
                    wait_outcome = self._wait_for_progress(rect=None)
                    if wait_outcome == "cancelled":
                        return
                    outcome_note = self._outcome_after_quality_rejection(
                        decision,
                        final_ocr_verification.reason,
                    )
                    continue
            self._emit_target_diagnostic(
                build_target_diagnostic(
                    decision=decision,
                    capture=final_capture,
                    candidates=final_candidates,
                    target=final_display_target,
                    quality=final_quality,
                    ocr=final_ocr_verification,
                    overlay_rect=final_rect,
                )
            )
            self._show_step(decision.instruction, final_rect)
            wait_outcome = self._wait_for_progress(rect=final_rect)
            self._set_active_rect(None)
            if wait_outcome == "cancelled":
                return
            self._clear_overlays()
            if wait_outcome == "clicked_inside" and self._sleep_with_cancel(POST_CLICK_SETTLE_SEC):
                return
            outcome_note = self._outcome_after_step(decision, wait_outcome)

        self._clear_overlays()
        self.chat_status.emit("")
        self.step_skipped.emit("Stopping the walkthrough — too many steps.")
        self.finished.emit("Stopped after too many steps.")

    def _show_step(self, instruction: str, rect: tuple[int, int, int, int]) -> None:
        x, y, width, height = rect
        self.highlight_show.emit(int(x), int(y), int(width), int(height), instruction)
        self.chat_status.emit(instruction)
        self._set_active_rect(rect)

    def _revalidate_target_on_current_screen(
        self,
        decision: "LiveHelpDecision",
        *,
        previous_target: TargetResolution,
        previous_capture: Any = None,
        previous_candidates: list[ControlCandidate] | None = None,
    ) -> tuple["Capture", list[ControlCandidate], TargetResolution]:
        self._clear_overlays(wait_for_flush=True)
        capture = self._capture_provider()
        candidates = self._collect_candidates(capture)
        if not candidates and previous_target.source in UIA_BACKED_TARGET_SOURCES:
            return (
                capture,
                candidates,
                replace(
                    previous_target,
                    rejected_reason="current screen recheck candidates unavailable",
                ),
            )
        target = resolve_help_target(
            decision,
            capture,
            candidates,
            snapper=self._snapper,
            clip_to_capture=False,
        )
        target = _guard_revalidated_target(
            decision=decision,
            capture=capture,
            candidates=candidates,
            previous_target=previous_target,
            previous_capture=previous_capture,
            previous_candidates=previous_candidates,
            target=target,
            snapper=self._snapper,
        )
        return capture, candidates, target

    def _collect_candidates(self, capture: "Capture") -> list[ControlCandidate]:
        run = self._run_context()
        cancelled = run.cancelled if run is not None else self._cancelled
        for attempt in range(CANDIDATE_EMPTY_RETRIES + 1):
            candidates = self._candidate_provider(capture)
            if candidates or attempt >= CANDIDATE_EMPTY_RETRIES or self._aborted():
                return candidates
            log.debug("Help candidate snapshot was empty; retrying before model prompt")
            if cancelled.wait(CANDIDATE_EMPTY_RETRY_SEC):
                return []
        return []

    def _wait_for_progress(self, rect: tuple[int, int, int, int] | None) -> str:
        run = self._run_context()
        cancelled = run.cancelled if run is not None else self._cancelled
        click_inside = run.click_inside if run is not None else self._click_inside_event
        check_now = run.check_now if run is not None else self._check_now_event
        click_inside.clear()
        check_now.clear()
        if rect is None:
            self._set_active_rect(None)
        deadline = time.monotonic() + IDLE_RECHECK_SEC
        while True:
            if cancelled.is_set():
                return "cancelled"
            if click_inside.is_set():
                return "clicked_inside"
            if check_now.is_set():
                check_now.clear()
                return "clicked_elsewhere"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "idle"
            cancelled.wait(min(remaining, 0.04))

    @staticmethod
    def _outcome_after_step(decision: "LiveHelpDecision", outcome: str) -> str:
        instruction = decision.instruction.strip().rstrip(".")
        expected = decision.expected_change.strip().rstrip(".")
        expected_line = (
            f' Expected visible change: "{expected}".'
            if expected
            else ""
        )
        if outcome == "clicked_inside":
            return (
                f'You suggested: "{instruction}". The user clicked the '
                "highlighted target."
                f"{expected_line} Look at the current screen and continue only "
                "from what is actually visible."
            )
        if outcome == "clicked_elsewhere":
            return (
                f'You suggested: "{instruction}". The user clicked somewhere '
                "else on the screen, not the highlighted target. Look at the "
                "current screen and either re-target or narrate to re-orient them."
            )
        return (
            f'You suggested: "{instruction}". The user has not clicked yet. '
            f"{expected_line} Look at the current screen — it may have changed on its own — "
            "and decide the next step."
        )

    @staticmethod
    def _outcome_after_narrate(outcome: str) -> str:
        if outcome in {"clicked_inside", "clicked_elsewhere"}:
            return "The user clicked. Continue from the current screen."
        return "Continue from the current screen."

    @staticmethod
    def _outcome_after_downgrade(
        decision: "LiveHelpDecision",
        reason: str = "target was rejected",
    ) -> str:
        instruction = decision.instruction.strip().rstrip(".")
        return (
            f'You suggested: "{instruction}". The target was not drawn because '
            f"{reason}. Emit a precise target around the actual clickable "
            "element only if you can identify it confidently — otherwise use narrate."
        )

    @staticmethod
    def _outcome_after_quality_rejection(
        decision: "LiveHelpDecision",
        reason: str,
    ) -> str:
        instruction = decision.instruction.strip().rstrip(".")
        return (
            f'You suggested: "{instruction}". The resolved target was not drawn '
            f"because {reason}. Re-check the screenshot and either emit a "
            "precise visible target with stronger evidence or use narrate."
        )

    @staticmethod
    def _outcome_after_helper_action(action: dict[str, Any]) -> str:
        name = str(action.get("name") or "").lower()
        if name == "launch_app":
            target = action.get("display_name") or safe_target_label(action.get("command"))
            return f"Helper just launched {target}. Continue from the new screen."
        if name == "open_url":
            target = safe_target_label(action.get("url"))
            return f"Helper just opened {target}. Continue from the new screen."
        if name == "key":
            return (
                f"Helper just pressed {action.get('keys')}. "
                "Continue from the new screen."
            )
        if name == "scroll":
            direction = action.get("direction") or "down"
            return f"Helper just scrolled {direction}. Continue from the new screen."
        return "Helper just ran a setup action. Continue from the new screen."

    def _end_walkthrough(self, message: str) -> None:
        self._clear_overlays()
        self.chat_message.emit(message)
        self.chat_status.emit("")
        self.finished.emit(message)

    def _clear_overlays(self, *, wait_for_flush: bool = False) -> None:
        self.ghost_clear.emit()
        self.highlight_clear.emit()
        self._set_active_rect(None)
        if wait_for_flush:
            self._flush_overlay_clear()

    def _flush_overlay_clear(self) -> None:
        if self._overlay_clear_barrier is None:
            return
        try:
            self._overlay_clear_barrier()
        except Exception:
            log.exception("Overlay clear barrier failed")
            return
        if OVERLAY_CLEAR_SETTLE_SEC > 0:
            run = self._run_context()
            if run is not None:
                run.cancelled.wait(OVERLAY_CLEAR_SETTLE_SEC)

    def _set_active_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        with self._rect_lock:
            self._active_rect = rect

    def _emit_target_diagnostic(self, payload: dict[str, Any]) -> None:
        try:
            log.info("Help target diagnostic: %s", json.dumps(payload, sort_keys=True))
        except Exception:
            log.info("Help target diagnostic: %r", payload)
        self.target_diagnostic.emit(payload)

    def _execute_helper_action(self, action: dict[str, Any]) -> None:
        name = str(action.get("name") or "").lower()
        try:
            if name == "launch_app":
                command = str(action.get("command") or "").strip()
                if command:
                    self._launch(command)
            elif name == "open_url":
                url = str(action.get("url") or "").strip()
                if url:
                    self._launch(url)
            elif name == "key":
                keys = str(action.get("keys") or "").strip()
                if keys:
                    self._controller.key(
                        keys,
                        description="Walkthrough setup.",
                        force_execute=True,
                    )
            elif name == "scroll":
                direction = str(action.get("direction") or "down").lower()
                dy = -700 if direction == "down" else 700
                self._controller.scroll(dy, description="Walkthrough scroll.")
        except Exception:
            log.exception(
                "Helper action failed: %s (%s)",
                name,
                safe_target_label(action.get("url") or action.get("command")),
            )

    @staticmethod
    def _launch(target: str) -> None:
        try:
            launch_target(target, allowed_commands=HELP_LAUNCH_COMMANDS)
        except (LaunchError, OSError, ValueError):
            log.exception("Launch failed: %s", safe_target_label(target))

    def _sleep_with_cancel(self, seconds: float) -> bool:
        run = self._run_context()
        if run is None or run.cancelled.wait(seconds):
            return True
        return self._aborted()

    def _aborted(self) -> bool:
        run = self._run_context()
        if run is None:
            if self._cancelled.is_set():
                return True
            abort = getattr(self._controller, "abort_controller", None)
            return bool(abort is not None and abort.is_aborted())
        if not self._is_current_run(run):
            return True
        if run.cancelled.is_set():
            return True
        abort = getattr(self._controller, "abort_controller", None)
        if abort is not None and abort.is_aborted():
            run.cancelled.set()
            return True
        return False
