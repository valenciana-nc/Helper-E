from __future__ import annotations

import io
import os
import threading
import time
import unittest
from typing import Any

from PIL import Image, ImageDraw
from PyQt6.QtWidgets import QApplication

from agent import LiveHelpDecision
from control_inventory import ControlCandidate, TargetResolution
from help_session import HelpSession
from ocr_text import OcrTextResult
from screen import Capture


def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _button_capture(button_rect: tuple[int, int, int, int] = (40, 50, 120, 32)) -> Capture:
    image = Image.new("RGB", (240, 160), color=(244, 246, 249))
    draw = ImageDraw.Draw(image)
    x, y, width, height = button_rect
    draw.rectangle((x, y, x + width, y + height), fill=(45, 100, 190), outline=(18, 48, 130), width=2)
    draw.rectangle((x + 8, y + 8, x + 32, y + height - 8), fill=(245, 248, 255))
    draw.line((x + 42, y + 10, x + width - 10, y + 10), fill=(245, 248, 255), width=3)
    draw.line((x + 42, y + 20, x + width - 28, y + 20), fill=(245, 248, 255), width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Capture(
        png_bytes=buffer.getvalue(),
        width=image.width,
        height=image.height,
        monitor_left=0,
        monitor_top=0,
        scale=1.0,
    )


def _checkbox_capture(
    checkbox_rect: tuple[int, int, int, int] = (40, 50, 24, 24),
    label: str = "Terms",
) -> Capture:
    image = Image.new("RGB", (240, 160), color=(244, 246, 249))
    draw = ImageDraw.Draw(image)
    x, y, width, height = checkbox_rect
    box_size = min(20, height)
    box_y = y + (height - box_size) // 2
    draw.rectangle((x, box_y, x + box_size, box_y + box_size), fill=(255, 255, 255), outline=(18, 48, 130), width=2)
    draw.line((x + 4, box_y + 10, x + 8, box_y + 15), fill=(45, 100, 190), width=2)
    draw.line((x + 8, box_y + 15, x + 16, box_y + 5), fill=(45, 100, 190), width=2)
    draw.text((x + width + 4, y + 4), label, fill=(18, 24, 38))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Capture(
        png_bytes=buffer.getvalue(),
        width=image.width,
        height=image.height,
        monitor_left=0,
        monitor_top=0,
        scale=1.0,
    )


def _dialog_ok_capture(
    title: str,
    body: str,
    button_rect: tuple[int, int, int, int] = (120, 110, 60, 28),
) -> Capture:
    image = Image.new("RGB", (240, 160), color=(238, 240, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 18, 224, 148), fill=(255, 255, 255), outline=(130, 136, 148), width=2)
    draw.text((28, 34), title, fill=(18, 24, 38))
    draw.text((28, 58), body, fill=(56, 63, 77))
    x, y, width, height = button_rect
    draw.rectangle((x, y, x + width, y + height), fill=(52, 103, 190), outline=(26, 60, 130), width=2)
    draw.text((x + 21, y + 8), "OK", fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Capture(
        png_bytes=buffer.getvalue(),
        width=image.width,
        height=image.height,
        monitor_left=0,
        monitor_top=0,
        scale=1.0,
    )


class _AbortController:
    def is_aborted(self) -> bool:
        return False


class _Controller:
    abort_controller = _AbortController()


class _ScriptedAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def plan_next_step(
        self,
        history: Any,
        *,
        control_candidates: list[ControlCandidate] | None = None,
        capture: Capture | None = None,
    ) -> LiveHelpDecision:
        self.calls.append(
            {
                "messages": [(msg.role, msg.text) for msg in history.conversation_messages()],
                "control_candidates": list(control_candidates or []),
                "capture": capture,
            }
        )
        if len(self.calls) == 1:
            return LiveHelpDecision(
                kind="step",
                instruction="Click Save changes.",
                expected_change="A saved confirmation appears.",
                target_id="c001",
                target_norm_x=780,
                target_norm_y=760,
                target_norm_width=80,
                target_norm_height=80,
            )
        return LiveHelpDecision(kind="done", message="Saved.")


class _WrongTargetIdAgent(_ScriptedAgent):
    def plan_next_step(
        self,
        history: Any,
        *,
        control_candidates: list[ControlCandidate] | None = None,
        capture: Capture | None = None,
    ) -> LiveHelpDecision:
        self.calls.append(
            {
                "messages": [(msg.role, msg.text) for msg in history.conversation_messages()],
                "control_candidates": list(control_candidates or []),
                "capture": capture,
            }
        )
        if len(self.calls) == 1:
            return LiveHelpDecision(
                kind="step",
                instruction="Click Save changes.",
                expected_change="A saved confirmation appears.",
                target_id="c002",
            )
        return LiveHelpDecision(kind="done", message="Saved.")


class _DoneAgent:
    def plan_next_step(
        self,
        history: Any,
        *,
        control_candidates: list[ControlCandidate] | None = None,
        capture: Capture | None = None,
    ) -> LiveHelpDecision:
        return LiveHelpDecision(kind="done", message="Done.")


class _ModelRectAgent:
    def __init__(self) -> None:
        self.calls = 0

    def plan_next_step(
        self,
        history: Any,
        *,
        control_candidates: list[ControlCandidate] | None = None,
        capture: Capture | None = None,
    ) -> LiveHelpDecision:
        self.calls += 1
        if self.calls == 1:
            return LiveHelpDecision(
                kind="step",
                instruction="Click this button.",
                expected_change="A saved confirmation appears.",
                target_norm_x=167,
                target_norm_y=313,
                target_norm_width=333,
                target_norm_height=200,
            )
        return LiveHelpDecision(kind="done", message="Done.")


class _OkAgent:
    def __init__(self) -> None:
        self.calls = 0

    def plan_next_step(
        self,
        history: Any,
        *,
        control_candidates: list[ControlCandidate] | None = None,
        capture: Capture | None = None,
    ) -> LiveHelpDecision:
        self.calls += 1
        if self.calls == 1:
            return LiveHelpDecision(
                kind="step",
                instruction="Click OK.",
                expected_change="The dialog closes.",
                target_id="ok",
            )
        return LiveHelpDecision(kind="done", message="Done.")


class _FakeOcrProvider:
    def __init__(self, result: OcrTextResult) -> None:
        self.result = result
        self.calls: list[tuple[Capture, tuple[int, int, int, int]]] = []

    def recognize_text(
        self,
        capture: Capture,
        rect: tuple[int, int, int, int],
    ) -> OcrTextResult:
        self.calls.append((capture, rect))
        return self.result


class _MutatingOcrProvider(_FakeOcrProvider):
    def __init__(self, result: OcrTextResult, on_call: Any) -> None:
        super().__init__(result)
        self._on_call = on_call

    def recognize_text(
        self,
        capture: Capture,
        rect: tuple[int, int, int, int],
    ) -> OcrTextResult:
        result = super().recognize_text(capture, rect)
        self._on_call()
        return result


class _SequencingOcrProvider:
    def __init__(self, results: list[OcrTextResult], on_first_call: Any | None = None) -> None:
        self.results = list(results)
        self.calls: list[tuple[Capture, tuple[int, int, int, int]]] = []
        self._on_first_call = on_first_call

    def recognize_text(
        self,
        capture: Capture,
        rect: tuple[int, int, int, int],
    ) -> OcrTextResult:
        self.calls.append((capture, rect))
        index = min(len(self.calls) - 1, max(0, len(self.results) - 1))
        result = self.results[index] if self.results else OcrTextResult(available=False)
        if len(self.calls) == 1 and self._on_first_call is not None:
            self._on_first_call()
        return result


class HelpSessionEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_ocr_env = os.environ.get("HELP_OCR_TEXT_VERIFY")
        os.environ["HELP_OCR_TEXT_VERIFY"] = "0"

    def tearDown(self) -> None:
        if self._previous_ocr_env is None:
            os.environ.pop("HELP_OCR_TEXT_VERIFY", None)
        else:
            os.environ["HELP_OCR_TEXT_VERIFY"] = self._previous_ocr_env

    def test_click_hit_margin_scales_with_target_size(self) -> None:
        from help_session import click_hit_margin

        self.assertEqual(click_hit_margin((100, 100, 32, 32)), 10)
        self.assertEqual(click_hit_margin((100, 100, 120, 32)), 10)
        self.assertEqual(click_hit_margin((100, 100, 300, 300)), 24)

    def test_small_target_click_just_outside_adaptive_margin_counts_elsewhere(self) -> None:
        _qt_app()
        session = HelpSession(
            _ScriptedAgent(),
            _Controller(),
            capture_provider=lambda: _button_capture(),
            candidate_provider=lambda _capture: [],
        )

        session._set_active_rect((100, 100, 32, 32))
        session.notify_user_click(143, 116)

        self.assertFalse(session._click_inside_event.is_set())
        self.assertTrue(session._check_now_event.is_set())

    def test_small_target_click_inside_adaptive_margin_counts_inside(self) -> None:
        _qt_app()
        session = HelpSession(
            _ScriptedAgent(),
            _Controller(),
            capture_provider=lambda: _button_capture(),
            candidate_provider=lambda _capture: [],
        )

        session._set_active_rect((100, 100, 32, 32))
        session.notify_user_click(141, 116)

        self.assertTrue(session._click_inside_event.is_set())
        self.assertFalse(session._check_now_event.is_set())

    def test_replacing_session_isolates_old_thread_state(self) -> None:
        _qt_app()
        first_entered = threading.Event()
        release_first = threading.Event()

        class _BlockingAgent:
            def __init__(self) -> None:
                self.calls = 0

            def plan_next_step(self, history: Any, **_kwargs: Any) -> LiveHelpDecision:
                self.calls += 1
                if self.calls == 1:
                    first_entered.set()
                    release_first.wait(timeout=2.0)
                return LiveHelpDecision(kind="done", message="Done.")

        session = HelpSession(
            _BlockingAgent(),  # type: ignore[arg-type]
            _Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: _button_capture(),
            candidate_provider=lambda _capture: [],
        )

        session.start("first")
        first_thread = session._thread
        self.assertIsNotNone(first_thread)
        self.assertTrue(first_entered.wait(timeout=1.0))

        session.start("second")
        second_thread = session._thread
        self.assertIsNotNone(second_thread)
        release_first.set()

        first_thread.join(timeout=1.0)  # type: ignore[union-attr]
        second_thread.join(timeout=1.0)  # type: ignore[union-attr]
        self.assertFalse(first_thread.is_alive())  # type: ignore[union-attr]
        self.assertFalse(second_thread.is_alive())  # type: ignore[union-attr]
        session.cancel()

    def test_help_session_waits_for_overlay_clear_barrier_before_capture(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        barrier_entered = threading.Event()
        barrier_release = threading.Event()
        capture_called = threading.Event()
        capture_saw_released: list[bool] = []

        def overlay_clear_barrier() -> None:
            barrier_entered.set()
            self.assertTrue(barrier_release.wait(timeout=2.0))

        def capture_provider() -> Capture:
            capture_saw_released.append(barrier_release.is_set())
            capture_called.set()
            return capture

        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=capture_provider,
            candidate_provider=lambda _capture: [],
            overlay_clear_barrier=overlay_clear_barrier,
        )
        finished: list[str] = []
        failed: list[str] = []
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        try:
            session.start("Help me.")
            self.assertTrue(barrier_entered.wait(timeout=1.0))
            time.sleep(0.08)
            app.processEvents()
            self.assertFalse(capture_called.is_set())
            barrier_release.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Done."])
        self.assertEqual(capture_saw_released, [True])

    def test_help_session_emits_candidate_highlight_and_click_outcome(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        candidate = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        agent = _ScriptedAgent()
        candidate_captures: list[Capture] = []

        def candidate_provider(seen_capture: Capture) -> list[ControlCandidate]:
            candidate_captures.append(seen_capture)
            return [candidate]

        def snapper(_rect: tuple[int, int, int, int], _instruction: str):
            raise AssertionError("target_id candidate evidence should resolve before snapper")

        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=candidate_provider,
            snapper=snapper,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        clears: list[bool] = []

        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))
        session.highlight_clear.connect(lambda: clears.append(True))

        click_sent = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if highlights and not click_sent:
                    x, y, width, height, _label = highlights[0]
                    time.sleep(0.05)
                    session.notify_user_click(x + width // 2, y + height // 2)
                    click_sent = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [(40, 50, 120, 32, "Click Save changes.")])
        self.assertGreaterEqual(len(clears), 2)

        self.assertGreaterEqual(len(diagnostics), 1)
        diagnostic = diagnostics[0]
        self.assertTrue(diagnostic["overlay"]["emitted"])
        self.assertEqual(diagnostic["overlay"]["rect"], (40, 50, 120, 32))
        self.assertEqual(diagnostic["resolution"]["source"], "target_id")
        self.assertEqual(diagnostic["resolution"]["target_id"], "c001")
        self.assertEqual(diagnostic["resolution"]["matched_text"], "Save changes saveButton")

        self.assertGreaterEqual(len(agent.calls), 2)
        self.assertEqual(agent.calls[0]["control_candidates"], [candidate])
        self.assertIs(agent.calls[0]["capture"], capture)
        self.assertTrue(all(seen is capture for seen in candidate_captures))

        followup_text = agent.calls[1]["messages"][-1][1]
        self.assertIn("The user clicked the highlighted target.", followup_text)
        self.assertIn('Expected visible change: "A saved confirmation appears".', followup_text)
        assistant_history = [
            text for role, text in agent.calls[1]["messages"] if role == "assistant"
        ]
        self.assertTrue(assistant_history)
        self.assertFalse(any("target_id=" in text for text in assistant_history))

    def test_help_session_ocr_unavailable_still_emits_candidate_highlight(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        candidate = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        ocr_provider = _FakeOcrProvider(OcrTextResult(available=False, error="OCR unavailable"))
        agent = _ScriptedAgent()
        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [candidate],
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []

        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        click_sent = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if highlights and not click_sent:
                    x, y, width, height, _label = highlights[0]
                    session.notify_user_click(x + width // 2, y + height // 2)
                    click_sent = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [(40, 50, 120, 32, "Click Save changes.")])
        self.assertEqual(len(ocr_provider.calls), 1)
        self.assertEqual(ocr_provider.calls[0][1], (40, 50, 120, 32))
        self.assertFalse(diagnostics[0]["ocr"]["available"])
        self.assertTrue(diagnostics[0]["overlay"]["emitted"])

    def test_help_session_ocr_disagreement_downgrades_before_highlight(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        candidate = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        ocr_provider = _FakeOcrProvider(OcrTextResult(text="Cancel", available=True))
        agent = _ScriptedAgent()
        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [candidate],
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []

        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [])
        self.assertEqual(len(ocr_provider.calls), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["overlay"]["rejected_reason"], "ocr text mismatch")
        self.assertEqual(diagnostics[0]["ocr"]["expected_text"], "Save changes")
        self.assertEqual(diagnostics[0]["ocr"]["recognized_text"], "Cancel")

    def test_help_session_available_blank_ocr_downgrades_before_highlight(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        candidate = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        ocr_provider = _FakeOcrProvider(OcrTextResult(text="", available=True))
        session = HelpSession(
            agent=_ScriptedAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [candidate],
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []

        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [])
        self.assertEqual(len(ocr_provider.calls), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["overlay"]["rejected_reason"], "ocr text missing")
        self.assertEqual(diagnostics[0]["ocr"]["expected_text"], "Save changes")
        self.assertEqual(diagnostics[0]["ocr"]["recognized_text"], "")

    def test_help_session_ocr_uses_state_label_evidence_rect_without_moving_overlay(self) -> None:
        app = _qt_app()
        capture = _checkbox_capture()
        checkbox = ControlCandidate(
            id="terms",
            text="Checked",
            control_type="checkbox",
            rect=(40, 52, 21, 21),
        )
        label = ControlCandidate(
            id="terms_label",
            text="Terms",
            control_type="text",
            rect=(64, 52, 80, 21),
        )
        ocr_provider = _FakeOcrProvider(OcrTextResult(text="Terms", available=True))

        class _TermsAgent(_ScriptedAgent):
            def plan_next_step(
                self,
                history: Any,
                *,
                control_candidates: list[ControlCandidate] | None = None,
                capture: Capture | None = None,
            ) -> LiveHelpDecision:
                self.calls.append(
                    {
                        "messages": [(msg.role, msg.text) for msg in history.conversation_messages()],
                        "control_candidates": list(control_candidates or []),
                        "capture": capture,
                    }
                )
                if len(self.calls) == 1:
                    return LiveHelpDecision(
                        kind="step",
                        instruction="Click the Terms checkbox.",
                        expected_change="Terms is toggled.",
                        target_id="terms",
                        target_norm_x=167,
                        target_norm_y=313,
                        target_norm_width=200,
                        target_norm_height=150,
                    )
                return LiveHelpDecision(kind="done", message="Done.")

        session = HelpSession(
            agent=_TermsAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [checkbox, label],
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []

        session.highlight_show.connect(
            lambda x, y, w, h, label_text: highlights.append((x, y, w, h, label_text))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        click_sent = False
        try:
            session.start("Help me accept the terms.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if highlights and not click_sent:
                    x, y, width, height, _label = highlights[0]
                    session.notify_user_click(x + width // 2, y + height // 2)
                    click_sent = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Done."])
        self.assertEqual(highlights, [(40, 52, 21, 21, "Click the Terms checkbox.")])
        self.assertEqual(len(ocr_provider.calls), 2)
        self.assertEqual(ocr_provider.calls[0][1], (40, 52, 104, 21))
        self.assertEqual(ocr_provider.calls[1][1], (40, 52, 104, 21))
        self.assertTrue(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["overlay"]["rect"], (40, 52, 21, 21))
        self.assertEqual(diagnostics[0]["ocr"]["expected_text"], "Terms")

    def test_final_pre_overlay_recheck_rejects_target_covered_after_ocr(self) -> None:
        app = _qt_app()
        clean_capture = _button_capture()
        covered_capture = _dialog_ok_capture(
            "Confirm changes",
            "This dialog covers the stale page button.",
            button_rect=(120, 110, 60, 28),
        )
        covered = False

        def mark_covered() -> None:
            nonlocal covered
            covered = True

        def capture_provider() -> Capture:
            return covered_capture if covered else clean_capture

        def candidate_provider(_capture: Capture) -> list[ControlCandidate]:
            save = ControlCandidate(
                id="c001",
                text="Save changes",
                control_type="button",
                rect=(40, 50, 120, 32),
                automation_id="saveButton",
                window_rank=1,
            )
            if not covered:
                return [save]
            return [
                ControlCandidate("dialog", "Confirm changes dialog", "window", (16, 18, 208, 130), window_rank=0),
                ControlCandidate("ok", "OK", "button", (120, 110, 60, 28), window_rank=0),
                save,
            ]

        ocr_provider = _MutatingOcrProvider(
            OcrTextResult(text="Save changes", available=True),
            mark_covered,
        )
        session = HelpSession(
            agent=_ScriptedAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=capture_provider,
            candidate_provider=candidate_provider,
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [])
        self.assertEqual(len(ocr_provider.calls), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(
            diagnostics[0]["overlay"]["rejected_reason"],
            "final pre-overlay recheck: current screen recheck target changed",
        )
        self.assertEqual(diagnostics[0]["ocr"]["recognized_text"], "Save changes")

    def test_final_coverage_gate_rejects_same_rank_menuitem_covering_target(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save changes",
            target_id="save",
        )
        candidates = [
            ControlCandidate("save", "Save changes", "button", (40, 50, 120, 32), window_rank=0),
            ControlCandidate("archive", "Archive", "menuitem", (35, 45, 132, 42), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(target, candidates),
            "target covered before overlay",
        )

    def test_final_coverage_gate_allows_same_rank_parent_menu_for_selected_menuitem(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="save",
        )
        candidates = [
            ControlCandidate("menu", "File menu", "menu", (20, 30, 180, 140), window_rank=0),
            ControlCandidate("save", "Save", "menuitem", (40, 50, 120, 32), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_rejects_new_same_rank_popup_surface_covering_target(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate("page_save", "Save", "button", (100, 100, 80, 32), window_rank=0),
        ]
        current_candidates = [
            previous_candidates[0],
            ControlCandidate("popup", "Settings popup", "window", (90, 90, 220, 150), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                current_candidates,
                previous_candidates=previous_candidates,
            ),
            "target covered before overlay",
        )

    def test_final_coverage_gate_rejects_existing_same_rank_popup_covering_background_target(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        candidates = [
            ControlCandidate(
                "page_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Dashboard",
                window_rank=0,
            ),
            ControlCandidate(
                "popup",
                "Settings popup",
                "window",
                (90, 90, 220, 150),
                window_title="Settings popup",
                window_rank=0,
            ),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "target covered before overlay",
        )

    def test_final_coverage_gate_allows_existing_same_rank_popup_parent(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(120, 110, 60, 28),
            confidence=1.0,
            source="target_id",
            matched_text="OK",
            target_id="ok",
        )
        candidates = [
            ControlCandidate(
                "popup",
                "Settings popup",
                "window",
                (90, 90, 220, 150),
                window_title="Settings popup",
                window_rank=0,
            ),
            ControlCandidate(
                "ok",
                "OK",
                "button",
                (120, 110, 60, 28),
                window_title="Settings popup",
                window_rank=0,
            ),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_allows_existing_same_window_shallower_surface_parent(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(120, 110, 60, 28),
            confidence=1.0,
            source="target_id",
            matched_text="Details",
            target_id="details",
        )
        candidates = [
            ControlCandidate(
                "pane",
                "Customer workspace",
                "pane",
                (90, 90, 220, 150),
                window_title="Dashboard",
                depth=1,
                window_rank=0,
            ),
            ControlCandidate(
                "details",
                "Details",
                "button",
                (120, 110, 60, 28),
                window_title="Dashboard",
                depth=2,
                window_rank=0,
            ),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_rejects_new_same_rank_dropdown_listitem_coverer(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate("page_save", "Save", "button", (100, 100, 80, 32), window_rank=0),
        ]
        candidates = previous_candidates + [
            ControlCandidate("suggestion", "Alice", "listitem", (95, 95, 95, 45), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=previous_candidates,
            ),
            "target covered before overlay",
        )

    def test_final_coverage_gate_allows_previous_row_owner_for_contained_action(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(730, 144, 60, 30),
            confidence=0.8,
            source="text_match",
            matched_text="Pay",
            target_id="pay",
        )
        candidates = [
            ControlCandidate("row", "INV-002 Beta Pending", "listitem", (20, 140, 800, 40), window_rank=0),
            ControlCandidate("pay", "Pay", "button", (730, 144, 60, 30), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_allows_previous_composite_parent_for_contained_button(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(286, 28, 24, 24),
            confidence=0.96,
            source="text_match",
            matched_text="Close",
            target_id="close",
        )
        candidates = [
            ControlCandidate("tab", "Docs - Project Plan", "tabitem", (100, 20, 220, 40), window_rank=0),
            ControlCandidate("close", "Close", "button", (286, 28, 24, 24), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_allows_same_rect_duplicate_control(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(600, 120, 80, 32),
            confidence=0.74,
            source="text_match",
            matched_text="Save primary-action",
            target_id="settings_save",
        )
        candidates = [
            ControlCandidate("cust_save", "Save", "button", (600, 120, 80, 32), window_title="Customers"),
            ControlCandidate("settings_save", "Save", "button", (600, 120, 80, 32), window_title="Settings"),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_rejects_foreground_same_rect_duplicate_control(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="background_save",
        )
        previous_candidates = [
            ControlCandidate(
                "background_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Page",
                window_rank=1,
            ),
        ]
        candidates = [
            ControlCandidate(
                "foreground_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Dialog",
                window_rank=0,
            ),
            previous_candidates[0],
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=previous_candidates,
            ),
            "target covered before overlay",
        )

    def test_final_coverage_gate_rejects_new_same_rank_same_rect_duplicate_control(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate(
                "page_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Page",
                window_rank=0,
            ),
        ]
        candidates = [
            previous_candidates[0],
            ControlCandidate(
                "dialog_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Dialog",
                window_rank=0,
            ),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=previous_candidates,
            ),
            "target covered before overlay",
        )

    def test_final_coverage_gate_rejects_moved_same_rank_same_rect_duplicate_control(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate(
                "page_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Page",
                window_rank=0,
            ),
            ControlCandidate(
                "dialog_save",
                "Save",
                "button",
                (420, 100, 80, 32),
                window_title="Dialog",
                window_rank=0,
            ),
        ]
        candidates = [
            previous_candidates[0],
            ControlCandidate(
                "dialog_save",
                "Save",
                "button",
                (100, 100, 80, 32),
                window_title="Dialog",
                window_rank=0,
            ),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=previous_candidates,
            ),
            "target covered before overlay",
        )

    def test_final_coverage_gate_allows_previous_row_owner_for_floating_menuitem(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(560, 124, 160, 28),
            confidence=0.58,
            source="candidate_snap",
            matched_text="Delete",
            target_id="delete",
        )
        candidates = [
            ControlCandidate("row", "Alice", "dataitem", (20, 80, 660, 110), window_rank=0),
            ControlCandidate("more", "More", "button", (560, 86, 64, 30), window_rank=0),
            ControlCandidate("delete", "Delete", "menuitem", (560, 124, 160, 28), window_rank=0),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_rejects_same_rank_form_and_structural_coverers(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate("page_save", "Save", "button", (100, 100, 80, 32), window_rank=0),
        ]
        coverers = (
            ControlCandidate("search", "Search", "edit", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("country", "Country", "combobox", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("enabled", "Enabled", "checkbox", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("weekly", "Weekly", "radiobutton", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("volume", "Volume", "slider", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("retries", "Retries", "spinner", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("row", "Result row", "dataitem", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("uia_row", "Result row", "row", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("tableitem", "Result row", "tableitem", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("tree", "Result tree item", "treeitem", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("cell", "Result cell", "cell", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("gridcell", "Result grid cell", "gridcell", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("datagridcell", "Result data grid cell", "datagridcell", (92, 94, 120, 42), window_rank=0),
            ControlCandidate("header", "Result header", "headeritem", (92, 94, 120, 42), window_rank=0),
        )

        for coverer in coverers:
            with self.subTest(control_type=coverer.control_type):
                self.assertEqual(
                    _foreground_candidate_covering_reason(
                        target,
                        previous_candidates + [coverer],
                        previous_candidates=previous_candidates,
                    ),
                    "target covered before overlay",
                )

    def test_final_coverage_gate_allows_same_rank_form_owner_for_contained_target(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(120, 110, 20, 20),
            confidence=1.0,
            source="target_id",
            matched_text="Checked",
            target_id="box",
        )
        candidates = [
            ControlCandidate(
                "field",
                "Notifications",
                "combobox",
                (100, 100, 180, 36),
                window_title="Preferences",
                depth=1,
                window_rank=0,
            ),
            ControlCandidate(
                "box",
                "Checked",
                "checkbox",
                (120, 110, 20, 20),
                window_title="Preferences",
                depth=2,
                window_rank=0,
            ),
        ]

        self.assertEqual(
            _foreground_candidate_covering_reason(
                target,
                candidates,
                previous_candidates=candidates,
            ),
            "",
        )

    def test_final_coverage_gate_rejects_new_same_rank_ordinary_named_surfaces(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate("page_save", "Save", "button", (100, 100, 80, 32), window_rank=0),
        ]
        surfaces = (
            ControlCandidate("toolbar", "Formatting", "toolbar", (90, 90, 220, 80), window_rank=0),
            ControlCandidate("menu", "File", "menu", (90, 90, 220, 80), window_rank=0),
            ControlCandidate("list", "Results", "list", (90, 90, 220, 80), window_rank=0),
            ControlCandidate("window", "Calculator", "window", (90, 90, 220, 80), window_rank=0),
        )

        for surface in surfaces:
            with self.subTest(control_type=surface.control_type, text=surface.text):
                self.assertEqual(
                    _foreground_candidate_covering_reason(
                        target,
                        previous_candidates + [surface],
                        previous_candidates=previous_candidates,
                    ),
                    "target covered before overlay",
                )

    def test_final_coverage_gate_rejects_new_same_rank_structural_popup_surfaces(self) -> None:
        from help_session import _foreground_candidate_covering_reason

        target = TargetResolution(
            rect=(100, 100, 80, 32),
            confidence=1.0,
            source="target_id",
            matched_text="Save",
            target_id="page_save",
        )
        previous_candidates = [
            ControlCandidate("page_save", "Save", "button", (100, 100, 80, 32), window_rank=0),
        ]
        surfaces = (
            ControlCandidate("pane", "Settings popup", "pane", (90, 90, 220, 150), window_rank=0),
            ControlCandidate("group", "Confirm changes dialog", "group", (90, 90, 220, 150), window_rank=0),
            ControlCandidate("list", "Suggestions", "list", (90, 90, 220, 150), window_rank=0),
            ControlCandidate("window", "Suggestions", "window", (90, 90, 220, 150), window_rank=0),
        )

        for surface in surfaces:
            with self.subTest(surface=surface.control_type, text=surface.text):
                self.assertEqual(
                    _foreground_candidate_covering_reason(
                        target,
                        previous_candidates + [surface],
                        previous_candidates=previous_candidates,
                    ),
                    "target covered before overlay",
                )

    def test_final_pre_overlay_full_path_rejects_same_rank_coverer_after_ocr(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        covered = False

        def mark_covered() -> None:
            nonlocal covered
            covered = True

        def candidate_provider(_capture: Capture) -> list[ControlCandidate]:
            save = ControlCandidate(
                id="c001",
                text="Save changes",
                control_type="button",
                rect=(40, 50, 120, 32),
                automation_id="saveButton",
                window_rank=0,
            )
            if not covered:
                return [save]
            return [
                save,
                ControlCandidate("menuitem", "Archive", "menuitem", (40, 45, 120, 24), window_rank=0),
            ]

        ocr_provider = _MutatingOcrProvider(
            OcrTextResult(text="Save changes", available=True),
            mark_covered,
        )
        session = HelpSession(
            agent=_ScriptedAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=candidate_provider,
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [])
        self.assertEqual(len(ocr_provider.calls), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(
            diagnostics[0]["overlay"]["rejected_reason"],
            "target covered before overlay",
        )

    def test_final_pre_overlay_ocr_rejects_text_changed_after_initial_ocr(self) -> None:
        app = _qt_app()
        clean_capture = _button_capture()
        changed_capture = _button_capture()
        changed = False

        def mark_changed() -> None:
            nonlocal changed
            changed = True

        def capture_provider() -> Capture:
            return changed_capture if changed else clean_capture

        candidate = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        ocr_provider = _SequencingOcrProvider(
            [
                OcrTextResult(text="Save changes", available=True),
                OcrTextResult(text="Delete changes", available=True),
            ],
            on_first_call=mark_changed,
        )
        session = HelpSession(
            agent=_ScriptedAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=capture_provider,
            candidate_provider=lambda _capture: [candidate],
            ocr_text_provider=ocr_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [])
        self.assertEqual(len(ocr_provider.calls), 2)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["overlay"]["rejected_reason"], "ocr partial text match")
        self.assertEqual(diagnostics[0]["ocr"]["expected_text"], "Save changes")
        self.assertEqual(diagnostics[0]["ocr"]["recognized_text"], "Delete changes")

    def test_help_session_retries_transient_empty_candidate_snapshot(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        candidate = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        agent = _ScriptedAgent()
        candidate_calls = 0

        def candidate_provider(_capture: Capture) -> list[ControlCandidate]:
            nonlocal candidate_calls
            candidate_calls += 1
            if candidate_calls == 1:
                return []
            return [candidate]

        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=candidate_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        click_sent = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if highlights and not click_sent:
                    x, y, width, height, _label = highlights[0]
                    session.notify_user_click(x + width // 2, y + height // 2)
                    click_sent = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertGreaterEqual(candidate_calls, 2)
        self.assertEqual(agent.calls[0]["control_candidates"], [candidate])
        self.assertEqual(highlights, [(40, 50, 120, 32, "Click Save changes.")])
        self.assertTrue(diagnostics[0]["overlay"]["emitted"])

    def test_help_session_revalidates_current_screen_before_emitting_highlight(self) -> None:
        app = _qt_app()
        first_capture = _button_capture((40, 50, 120, 32))
        current_capture = _button_capture((82, 64, 120, 32))
        capture_calls: list[Capture] = []
        agent = _ScriptedAgent()

        def capture_provider() -> Capture:
            capture = first_capture if not capture_calls else current_capture
            capture_calls.append(capture)
            return capture

        def candidate_provider(seen_capture: Capture) -> list[ControlCandidate]:
            rect = (40, 50, 120, 32) if seen_capture is first_capture else (82, 64, 120, 32)
            return [
                ControlCandidate(
                    id="c001",
                    text="Save changes",
                    control_type="button",
                    rect=rect,
                    automation_id="saveButton",
                )
            ]

        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=capture_provider,
            candidate_provider=candidate_provider,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        click_sent = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if highlights and not click_sent:
                    x, y, width, height, _label = highlights[0]
                    time.sleep(0.05)
                    session.notify_user_click(x + width // 2, y + height // 2)
                    click_sent = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertGreaterEqual(len(capture_calls), 2)
        self.assertEqual(highlights, [(82, 64, 120, 32, "Click Save changes.")])
        self.assertGreaterEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["resolution"]["rect"], (82, 64, 120, 32))

    def test_current_screen_recheck_rejects_stale_target_id_that_jumps_far(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [
                ControlCandidate("c001", "Save changes", "button", (170, 50, 120, 32)),
            ],
        )
        previous_target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=0.9,
            source="target_id",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
            target_id="c001",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_moderate_target_id_drift_when_text_confirms(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        rect = (82, 64, 120, 32)
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [
                ControlCandidate("c001", "Save changes", "button", rect),
            ],
        )
        previous_target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=0.9,
            source="target_id",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
            target_id="c001",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertFalse(target.rejected_reason)

    def test_current_screen_recheck_rejects_reused_row_action_context_change(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        current_candidates = [
            ControlCandidate("row_0", "Globex account", "dataitem", (20, 100, 700, 42)),
            ControlCandidate("approve", "Approve", "button", (600, 106, 80, 30)),
        ]
        previous_candidates = [
            ControlCandidate("row_0", "Acme account", "dataitem", (20, 100, 700, 42)),
            ControlCandidate("approve", "Approve", "button", (600, 106, 80, 30)),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=(600, 106, 80, 30),
            confidence=0.9,
            source="target_id",
            matched_text="Approve",
            target_id="approve",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Approve.",
            target_id="approve",
            target_norm_x=600,
            target_norm_y=106,
            target_norm_width=80,
            target_norm_height=30,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_reused_row_action_same_context(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        candidates = [
            ControlCandidate("row_0", "Acme account", "dataitem", (20, 100, 700, 42)),
            ControlCandidate("approve", "Approve", "button", (600, 106, 80, 30)),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: candidates,
        )
        previous_target = TargetResolution(
            rect=(600, 106, 80, 30),
            confidence=0.9,
            source="target_id",
            matched_text="Approve",
            target_id="approve",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Approve.",
            target_id="approve",
            target_norm_x=600,
            target_norm_y=106,
            target_norm_width=80,
            target_norm_height=30,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertFalse(target.rejected_reason)

    def test_current_screen_recheck_rejects_rebound_row_action_id_context_change(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        current_candidates = [
            ControlCandidate("row_1", "Globex account", "dataitem", (20, 100, 700, 42)),
            ControlCandidate("approve_new", "Approve", "button", (600, 106, 80, 30)),
        ]
        previous_candidates = [
            ControlCandidate("row_0", "Acme account", "dataitem", (20, 100, 700, 42)),
            ControlCandidate("approve_old", "Approve", "button", (600, 106, 80, 30)),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=(600, 106, 80, 30),
            confidence=0.9,
            source="target_id",
            matched_text="Approve",
            target_id="approve_old",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Approve.",
            target_id="approve_old",
            target_norm_x=600,
            target_norm_y=106,
            target_norm_width=80,
            target_norm_height=30,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "text_match")
        self.assertEqual(target.target_id, "approve_new")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_reused_action_identity_disappears(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        rect = (40, 50, 120, 32)
        current_candidates = [
            ControlCandidate("c001", "", "button", rect),
        ]
        previous_candidates = [
            ControlCandidate("c001", "Save changes", "button", rect),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
            target_id="c001",
            target_norm_x=167,
            target_norm_y=313,
            target_norm_width=500,
            target_norm_height=200,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_reused_action_same_identity(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        rect = (40, 50, 120, 32)
        current_candidates = [
            ControlCandidate("c001", "Save", "button", rect),
        ]
        previous_candidates = [
            ControlCandidate("c001", "Save changes", "button", rect),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
            target_id="c001",
            target_norm_x=167,
            target_norm_y=313,
            target_norm_width=500,
            target_norm_height=200,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertFalse(target.rejected_reason)

    def test_current_screen_recheck_rejects_action_context_from_other_window_rank(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        capture = _button_capture(button_rect=(200, 110, 60, 28))
        rect = (200, 110, 60, 28)
        previous_candidates = [
            ControlCandidate(
                "row",
                "Acme invoice",
                "dataitem",
                (20, 100, 260, 48),
                window_title="Billing",
                window_rank=0,
            ),
            ControlCandidate(
                "pay",
                "Pay",
                "button",
                rect,
                window_title="Billing",
                window_rank=0,
            ),
        ]
        current_candidates = [
            ControlCandidate(
                "row",
                "Acme invoice",
                "dataitem",
                (20, 100, 260, 48),
                window_title="Other App",
                window_rank=1,
            ),
            ControlCandidate(
                "pay",
                "Pay",
                "button",
                rect,
                window_title="Billing",
                window_rank=0,
            ),
        ]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Pay",
            target_id="pay",
        )
        target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Pay",
            target_id="pay",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Pay for Acme invoice.",
            target_id="pay",
            target_norm_x=667,
            target_norm_y=550,
            target_norm_width=200,
            target_norm_height=140,
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=capture,
            candidates=current_candidates,
            previous_target=previous_target,
            previous_capture=capture,
            previous_candidates=previous_candidates,
            target=target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_generic_action_when_visual_context_stable(self) -> None:
        app = _qt_app()
        rect = (120, 110, 60, 28)
        capture = _dialog_ok_capture(
            "Delete customer?",
            "This will remove Acme from the workspace.",
            button_rect=rect,
        )
        candidates = [
            ControlCandidate("ok", "OK", "button", rect),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: candidates,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="OK",
            target_id="ok",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click OK.",
            target_id="ok",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_capture=capture,
                previous_candidates=candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertFalse(target.rejected_reason)

    def test_current_screen_recheck_allows_specific_action_when_visual_context_stable(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 70, 28)
        capture = _dialog_ok_capture(
            "Customer details",
            "Showing Acme account details.",
            button_rect=rect,
        )
        candidates = [ControlCandidate("c001", "Details", "button", rect)]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        current_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Details.",
            target_id="c001",
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=capture,
            candidates=candidates,
            previous_target=previous_target,
            previous_capture=capture,
            previous_candidates=candidates,
            target=current_target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertFalse(guarded.rejected_reason)

    def test_current_screen_recheck_rejects_specific_action_reused_id_when_visual_context_changes(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 70, 28)
        previous_capture = _dialog_ok_capture(
            "Customer details",
            "Showing Acme account details.",
            button_rect=rect,
        )
        current_capture = _dialog_ok_capture(
            "Customer details",
            "Showing Globex account details.",
            button_rect=rect,
        )
        previous_candidates = [ControlCandidate("c001", "Details", "button", rect)]
        current_candidates = [ControlCandidate("c001", "Details", "button", rect)]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        current_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Details.",
            target_id="c001",
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=current_capture,
            candidates=current_candidates,
            previous_target=previous_target,
            previous_capture=previous_capture,
            previous_candidates=previous_candidates,
            target=current_target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_specific_action_when_context_text_stable(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 70, 28)
        capture = _dialog_ok_capture(
            "Customer details",
            "Showing Acme customer details.",
            button_rect=rect,
        )
        candidates = [
            ControlCandidate("dialog", "Customer dialog", "window", (16, 18, 208, 130), depth=0),
            ControlCandidate("body", "Showing Acme customer details.", "statictext", (28, 58, 170, 18), depth=1),
            ControlCandidate("c001", "Details", "button", rect, depth=1),
        ]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        current_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Details.",
            target_id="c001",
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=capture,
            candidates=candidates,
            previous_target=previous_target,
            previous_capture=capture,
            previous_candidates=candidates,
            target=current_target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertFalse(guarded.rejected_reason)

    def test_current_screen_recheck_rejects_specific_action_labels_when_context_text_changes(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 70, 28)
        previous_capture = _dialog_ok_capture(
            "Customer details",
            "Showing Acme customer details.",
            button_rect=rect,
        )
        current_capture = _dialog_ok_capture(
            "Customer details",
            "Showing Globex project details.",
            button_rect=rect,
        )

        for label in ("Details", "Open", "Edit"):
            with self.subTest(label=label):
                previous_candidates = [
                    ControlCandidate("dialog", "Customer dialog", "window", (16, 18, 208, 130), depth=0),
                    ControlCandidate(
                        "body",
                        "Showing Acme customer details.",
                        "statictext",
                        (28, 58, 170, 18),
                        depth=1,
                    ),
                    ControlCandidate("c001", label, "button", rect, depth=1),
                ]
                current_candidates = [
                    ControlCandidate("dialog", "Customer dialog", "window", (16, 18, 208, 130), depth=0),
                    ControlCandidate(
                        "body",
                        "Showing Globex project details.",
                        "statictext",
                        (28, 58, 170, 18),
                        depth=1,
                    ),
                    ControlCandidate("c001", label, "button", rect, depth=1),
                ]
                previous_target = TargetResolution(
                    rect=rect,
                    confidence=0.9,
                    source="target_id",
                    matched_text=label,
                    target_id="c001",
                )
                current_target = TargetResolution(
                    rect=rect,
                    confidence=0.9,
                    source="target_id",
                    matched_text=label,
                    target_id="c001",
                )
                decision = LiveHelpDecision(
                    kind="step",
                    instruction=f"Click {label}.",
                    target_id="c001",
                    target_norm_x=625,
                    target_norm_y=712,
                    target_norm_width=292,
                    target_norm_height=175,
                )

                guarded = _guard_revalidated_target(
                    decision=decision,
                    capture=current_capture,
                    candidates=current_candidates,
                    previous_target=previous_target,
                    previous_capture=previous_capture,
                    previous_candidates=previous_candidates,
                    target=current_target,
                    snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
                )

                self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_specific_action_when_weak_dialog_context_changes(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 70, 28)
        previous_candidates = [
            ControlCandidate("dialog", "Acme dialog", "window", (16, 18, 208, 130), depth=0),
            ControlCandidate("c001", "Open", "button", rect, depth=1),
        ]
        current_candidates = [
            ControlCandidate("dialog", "Globex dialog", "window", (16, 18, 208, 130), depth=0),
            ControlCandidate("c001", "Open", "button", rect, depth=1),
        ]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Open",
            target_id="c001",
        )
        current_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Open",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Open.",
            target_id="c001",
            target_norm_x=625,
            target_norm_y=712,
            target_norm_width=292,
            target_norm_height=175,
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=_dialog_ok_capture("Globex dialog", "", button_rect=rect),
            candidates=current_candidates,
            previous_target=previous_target,
            previous_capture=_dialog_ok_capture("Acme dialog", "", button_rect=rect),
            previous_candidates=previous_candidates,
            target=current_target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_specific_action_when_context_disappears(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 70, 28)
        previous_candidates = [
            ControlCandidate("row", "Acme customer", "dataitem", (82, 86, 150, 70), depth=0),
            ControlCandidate("c001", "Edit", "button", rect, depth=1),
        ]
        current_candidates = [
            ControlCandidate("c001", "Edit", "button", rect, depth=1),
        ]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Edit",
            target_id="c001",
        )
        current_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Edit",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Edit.",
            target_id="c001",
            target_norm_x=625,
            target_norm_y=712,
            target_norm_width=292,
            target_norm_height=175,
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=_button_capture(button_rect=rect),
            candidates=current_candidates,
            previous_target=previous_target,
            previous_capture=_button_capture(button_rect=rect),
            previous_candidates=previous_candidates,
            target=current_target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_help_session_downgrades_generic_action_when_visual_context_changes(self) -> None:
        app = _qt_app()
        rect = (120, 110, 60, 28)
        first_capture = _dialog_ok_capture(
            "Delete customer?",
            "This will remove Acme from the workspace.",
            button_rect=rect,
        )
        current_capture = _dialog_ok_capture(
            "Archive project?",
            "This will hide Project Orion from active lists.",
            button_rect=rect,
        )
        capture_calls: list[Capture] = []

        def capture_provider() -> Capture:
            capture = first_capture if not capture_calls else current_capture
            capture_calls.append(capture)
            return capture

        candidates = [
            ControlCandidate("ok", "OK", "button", rect),
        ]
        session = HelpSession(
            agent=_OkAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=capture_provider,
            candidate_provider=lambda _capture: candidates,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me confirm this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Done."])
        self.assertGreaterEqual(len(capture_calls), 2)
        self.assertEqual(highlights, [])
        self.assertGreaterEqual(len(diagnostics), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["resolution"]["source"], "target_id")
        self.assertEqual(
            diagnostics[0]["resolution"]["rejected_reason"],
            "current screen recheck target changed",
        )

    def test_help_session_downgrades_generic_action_with_weak_dialog_context_when_visual_changes(self) -> None:
        app = _qt_app()
        rect = (120, 110, 60, 28)
        dialog_rect = (16, 18, 208, 130)
        first_capture = _dialog_ok_capture(
            "Delete customer?",
            "This will remove Acme from the workspace.",
            button_rect=rect,
        )
        current_capture = _dialog_ok_capture(
            "Archive project?",
            "This will hide Project Orion from active lists.",
            button_rect=rect,
        )
        capture_calls: list[Capture] = []

        def capture_provider() -> Capture:
            capture = first_capture if not capture_calls else current_capture
            capture_calls.append(capture)
            return capture

        candidates = [
            ControlCandidate("dialog", "Dialog", "window", dialog_rect),
            ControlCandidate("ok", "OK", "button", rect),
        ]
        session = HelpSession(
            agent=_OkAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=capture_provider,
            candidate_provider=lambda _capture: candidates,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me confirm this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Done."])
        self.assertGreaterEqual(len(capture_calls), 2)
        self.assertEqual(highlights, [])
        self.assertGreaterEqual(len(diagnostics), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["resolution"]["source"], "target_id")
        self.assertEqual(
            diagnostics[0]["resolution"]["rejected_reason"],
            "current screen recheck target changed",
        )

    def test_current_screen_recheck_rejects_reused_control_identity_change(self) -> None:
        app = _qt_app()
        capture = _button_capture(button_rect=(40, 50, 80, 32))
        rect = (40, 50, 80, 32)
        cases = [
            ("Click this checkbox.", "checkbox", "Notifications", "Dark mode"),
            ("Click this combobox.", "combobox", "Country", "Language"),
            ("Click this edit control.", "edit", "Email", "Phone"),
            ("Select this radio option.", "radiobutton", "Weekly", "Daily"),
            ("Adjust this slider.", "slider", "Volume", "Brightness"),
            ("Adjust this spinner.", "spinner", "Retries", "Timeout"),
        ]

        for instruction, control_type, previous_text, current_text in cases:
            with self.subTest(control_type=control_type):
                current_candidates = [
                    ControlCandidate("setting", current_text, control_type, rect),
                ]
                previous_candidates = [
                    ControlCandidate("setting", previous_text, control_type, rect),
                ]
                session = HelpSession(
                    agent=_DoneAgent(),  # type: ignore[arg-type]
                    controller=_Controller(),  # type: ignore[arg-type]
                    capture_provider=lambda: capture,
                    candidate_provider=lambda _capture, items=current_candidates: items,
                )
                previous_target = TargetResolution(
                    rect=rect,
                    confidence=0.9,
                    source="target_id",
                    matched_text=previous_text,
                    target_id="setting",
                )
                decision = LiveHelpDecision(
                    kind="step",
                    instruction=instruction,
                    target_id="setting",
                    target_norm_x=167,
                    target_norm_y=313,
                    target_norm_width=333,
                    target_norm_height=200,
                )

                try:
                    _capture, _candidates, target = session._revalidate_target_on_current_screen(
                        decision,
                        previous_target=previous_target,
                        previous_candidates=previous_candidates,
                    )
                finally:
                    session.deleteLater()
                    app.processEvents()

                self.assertEqual(target.source, "target_id")
                self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_reused_control_same_identity(self) -> None:
        app = _qt_app()
        capture = _button_capture(button_rect=(40, 50, 80, 32))
        rect = (40, 50, 80, 32)
        current_candidates = [
            ControlCandidate("setting", "Notifications", "checkbox", rect),
        ]
        previous_candidates = [
            ControlCandidate("setting", "Notifications", "checkbox", rect),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Notifications",
            target_id="setting",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click this checkbox.",
            target_id="setting",
            target_norm_x=167,
            target_norm_y=313,
            target_norm_width=333,
            target_norm_height=200,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertFalse(target.rejected_reason)

    def test_current_screen_recheck_rejects_tabular_context_from_other_window_rank(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        capture = _button_capture(button_rect=(120, 110, 60, 24))
        rect = (120, 110, 60, 24)
        previous_candidates = [
            ControlCandidate("row", "Acme", "dataitem", (20, 100, 220, 44), window_title="Billing", window_rank=0),
            ControlCandidate("header", "Status", "headeritem", (120, 70, 80, 24), window_title="Billing", window_rank=0),
            ControlCandidate("cell", "Active", "cell", rect, window_title="Billing", window_rank=0),
        ]
        current_candidates = [
            ControlCandidate("row", "Acme", "dataitem", (20, 100, 220, 44), window_title="Other App", window_rank=1),
            ControlCandidate("header", "Status", "headeritem", (120, 70, 80, 24), window_title="Other App", window_rank=1),
            ControlCandidate("cell", "Active", "cell", rect, window_title="Billing", window_rank=0),
        ]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Active",
            target_id="cell",
        )
        target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Active",
            target_id="cell",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click the Status cell for Acme.",
            target_id="cell",
            target_norm_x=500,
            target_norm_y=688,
            target_norm_width=250,
            target_norm_height=150,
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=capture,
            candidates=current_candidates,
            previous_target=previous_target,
            previous_capture=capture,
            previous_candidates=previous_candidates,
            target=target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_control_section_context_from_other_window_rank(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        capture = _button_capture(button_rect=(120, 110, 24, 24))
        rect = (120, 110, 24, 24)
        previous_candidates = [
            ControlCandidate(
                "section",
                "Billing settings",
                "group",
                (90, 90, 180, 80),
                window_title="Billing",
                window_rank=0,
            ),
            ControlCandidate(
                "setting",
                "Enabled",
                "checkbox",
                rect,
                window_title="Billing",
                window_rank=0,
            ),
        ]
        current_candidates = [
            ControlCandidate(
                "section",
                "Billing settings",
                "group",
                (90, 90, 180, 80),
                window_title="Other App",
                window_rank=1,
            ),
            ControlCandidate(
                "setting",
                "Enabled",
                "checkbox",
                rect,
                window_title="Billing",
                window_rank=0,
            ),
        ]
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Enabled",
            target_id="setting",
        )
        target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Enabled",
            target_id="setting",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Enabled in Billing settings.",
            target_id="setting",
            target_norm_x=500,
            target_norm_y=688,
            target_norm_width=100,
            target_norm_height=150,
        )

        guarded = _guard_revalidated_target(
            decision=decision,
            capture=capture,
            candidates=current_candidates,
            previous_target=previous_target,
            previous_capture=capture,
            previous_candidates=previous_candidates,
            target=target,
            snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
        )

        self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_reused_control_section_automation_context_change(self) -> None:
        app = _qt_app()
        capture = _button_capture(button_rect=(40, 50, 80, 32))
        rect = (120, 110, 24, 24)
        previous_candidates = [
            ControlCandidate("section", "", "group", (90, 90, 180, 80), automation_id="shipping_section"),
            ControlCandidate("setting", "Enabled", "checkbox", rect),
        ]
        current_candidates = [
            ControlCandidate("section", "", "group", (90, 90, 180, 80), automation_id="billing_section"),
            ControlCandidate("setting", "Enabled", "checkbox", rect),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.9,
            source="target_id",
            matched_text="Enabled",
            target_id="setting",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click this checkbox.",
            target_id="setting",
            target_norm_x=500,
            target_norm_y=688,
            target_norm_width=100,
            target_norm_height=150,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_input_control_window_context_swap_without_decision_target_id(self) -> None:
        from help_session import _guard_revalidated_target
        from rect_snap import SnapResult

        rect = (120, 110, 80, 28)
        previous_capture = _dialog_ok_capture(
            "Billing settings",
            "Enable invoice preferences.",
            button_rect=rect,
        )
        current_capture = _dialog_ok_capture(
            "Shipping settings",
            "Enable shipment preferences.",
            button_rect=rect,
        )

        cases = (
            ("checkbox", "Checked", "Click Enable."),
            ("radiobutton", "Selected", "Select On."),
            ("edit", "", "Type in this field."),
            ("combobox", "On", "Open this dropdown."),
            ("slider", "50", "Adjust this slider."),
            ("spinner", "1", "Adjust this spinner."),
        )
        for control_type, text, instruction in cases:
            with self.subTest(control_type=control_type):
                previous_candidates = [
                    ControlCandidate("dialog", "Billing settings dialog", "window", (16, 18, 208, 140)),
                    ControlCandidate("setting", text, control_type, rect),
                ]
                current_candidates = [
                    ControlCandidate("dialog", "Shipping settings dialog", "window", (16, 18, 208, 140)),
                    ControlCandidate("setting", text, control_type, rect),
                ]
                previous_target = TargetResolution(
                    rect=rect,
                    confidence=0.9,
                    source="text_match",
                    matched_text=text,
                    target_id="setting",
                )
                current_target = TargetResolution(
                    rect=rect,
                    confidence=0.9,
                    source="text_match",
                    matched_text=text,
                    target_id="setting",
                )
                decision = LiveHelpDecision(kind="step", instruction=instruction)

                guarded = _guard_revalidated_target(
                    decision=decision,
                    capture=current_capture,
                    candidates=current_candidates,
                    previous_target=previous_target,
                    previous_capture=previous_capture,
                    previous_candidates=previous_candidates,
                    target=current_target,
                    snapper=lambda rect, _instruction: SnapResult(rect=rect, confidence=0.0, source="model"),
                )

                self.assertEqual(guarded.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_tiny_target_id_zero_overlap_drift(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [
                ControlCandidate("c001", "Info", "button", (48, 50, 8, 8)),
            ],
        )
        previous_target = TargetResolution(
            rect=(40, 50, 8, 8),
            confidence=0.9,
            source="target_id",
            matched_text="Info",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Info.",
            target_id="c001",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=[
                    ControlCandidate("c001", "Info", "button", (40, 50, 8, 8)),
                ],
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_nearby_nonoverlapping_target_id_rebind(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [
                ControlCandidate("c001", "Save changes", "button", (40, 95, 120, 32)),
            ],
        )
        previous_target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=0.9,
            source="target_id",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
            target_id="c001",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "target_id")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_text_match_that_jumps_far(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [
                ControlCandidate("c002", "Save changes", "button", (220, 50, 120, 32)),
            ],
        )
        previous_target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=0.9,
            source="text_match",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "text_match")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_allows_moderate_text_match_drift(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        rect = (82, 64, 120, 32)
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [
                ControlCandidate("c002", "Save changes", "button", rect),
            ],
        )
        previous_target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=0.9,
            source="text_match",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "text_match")
        self.assertFalse(target.rejected_reason)

    def test_current_screen_recheck_rejects_stale_target_id_replaced_by_nearby_text_match(self) -> None:
        app = _qt_app()
        capture = _button_capture(button_rect=(80, 80, 70, 28))
        current_candidates = [
            ControlCandidate("c002", "Details", "button", (80, 80, 70, 28)),
        ]
        previous_candidates = [
            ControlCandidate("c001", "Details", "button", (80, 60, 70, 28)),
        ]
        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: current_candidates,
        )
        previous_target = TargetResolution(
            rect=(80, 60, 70, 28),
            confidence=0.9,
            source="target_id",
            matched_text="Details",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Details.",
            target_id="c001",
            target_norm_x=333,
            target_norm_y=375,
            target_norm_width=292,
            target_norm_height=175,
        )

        try:
            _capture, _candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
                previous_candidates=previous_candidates,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(target.source, "text_match")
        self.assertEqual(target.target_id, "c002")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_current_screen_recheck_rejects_empty_candidates_for_candidate_backed_target(self) -> None:
        app = _qt_app()
        capture = _button_capture()

        def snapper(_rect: tuple[int, int, int, int], _instruction: str):
            raise AssertionError("model fallback should not run when recheck candidates vanish")

        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [],
            snapper=snapper,
        )
        previous_target = TargetResolution(
            rect=(40, 50, 120, 32),
            confidence=0.9,
            source="candidate_snap",
            matched_text="Save changes",
            target_id="c001",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click this button.",
            target_norm_x=166,
            target_norm_y=312,
            target_norm_width=500,
            target_norm_height=200,
        )

        try:
            _capture, candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(candidates, [])
        self.assertEqual(target.source, "candidate_snap")
        self.assertEqual(
            target.rejected_reason,
            "current screen recheck candidates unavailable",
        )

    def test_current_screen_recheck_rejects_model_only_target_without_fresh_evidence(self) -> None:
        from rect_snap import SnapResult

        app = _qt_app()
        rect = (40, 50, 80, 32)
        capture = _button_capture(button_rect=rect)

        def snapper(_rect: tuple[int, int, int, int], _instruction: str):
            return SnapResult(rect=rect, confidence=0.0, source="model")

        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [],
            snapper=snapper,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.0,
            source="model",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click this button.",
            target_norm_x=167,
            target_norm_y=313,
            target_norm_width=333,
            target_norm_height=200,
        )

        try:
            _capture, candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(candidates, [])
        self.assertEqual(target.source, "model")
        self.assertEqual(target.rejected_reason, "current screen recheck target changed")

    def test_help_session_downgrades_model_only_recheck_without_fresh_evidence(self) -> None:
        from rect_snap import SnapResult

        app = _qt_app()
        rect = (40, 50, 80, 32)
        capture = _button_capture(button_rect=rect)
        agent = _ModelRectAgent()

        def snapper(_rect: tuple[int, int, int, int], _instruction: str):
            return SnapResult(rect=rect, confidence=0.0, source="model")

        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [],
            snapper=snapper,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []
        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        advanced = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if diagnostics and not advanced:
                    session.notify_user_click(5, 5)
                    advanced = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Done."])
        self.assertEqual(highlights, [])
        self.assertGreaterEqual(len(diagnostics), 1)
        self.assertFalse(diagnostics[0]["overlay"]["emitted"])
        self.assertEqual(diagnostics[0]["resolution"]["source"], "model")
        self.assertEqual(
            diagnostics[0]["resolution"]["rejected_reason"],
            "current screen recheck target changed",
        )

    def test_current_screen_recheck_allows_model_origin_when_fresh_uia_snap_confirms(self) -> None:
        from rect_snap import SnapResult

        app = _qt_app()
        rect = (40, 50, 80, 32)
        capture = _button_capture(button_rect=rect)

        def snapper(_rect: tuple[int, int, int, int], _instruction: str):
            return SnapResult(
                rect=rect,
                confidence=0.9,
                source="uia",
                matched_text="Save changes",
            )

        session = HelpSession(
            agent=_DoneAgent(),  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [],
            snapper=snapper,
        )
        previous_target = TargetResolution(
            rect=rect,
            confidence=0.0,
            source="model",
        )
        decision = LiveHelpDecision(
            kind="step",
            instruction="Click Save changes.",
            target_norm_x=167,
            target_norm_y=313,
            target_norm_width=333,
            target_norm_height=200,
        )

        try:
            _capture, candidates, target = session._revalidate_target_on_current_screen(
                decision,
                previous_target=previous_target,
            )
        finally:
            session.deleteLater()
            app.processEvents()

        self.assertEqual(candidates, [])
        self.assertEqual(target.source, "snap")
        self.assertFalse(target.rejected_reason)

    def test_help_session_recovers_wrong_target_id_before_emitting_highlight(self) -> None:
        app = _qt_app()
        capture = _button_capture()
        save = ControlCandidate(
            id="c001",
            text="Save changes",
            control_type="button",
            rect=(40, 50, 120, 32),
            automation_id="saveButton",
        )
        cancel = ControlCandidate(
            id="c002",
            text="Cancel",
            control_type="button",
            rect=(170, 50, 60, 32),
            automation_id="cancelButton",
        )
        agent = _WrongTargetIdAgent()

        def snapper(_rect: tuple[int, int, int, int], _instruction: str):
            raise AssertionError("wrong target_id should recover from current candidates")

        session = HelpSession(
            agent=agent,  # type: ignore[arg-type]
            controller=_Controller(),  # type: ignore[arg-type]
            capture_provider=lambda: capture,
            candidate_provider=lambda _capture: [save, cancel],
            snapper=snapper,
        )
        highlights: list[tuple[int, int, int, int, str]] = []
        diagnostics: list[dict[str, Any]] = []
        finished: list[str] = []
        failed: list[str] = []

        session.highlight_show.connect(
            lambda x, y, w, h, label: highlights.append((x, y, w, h, label))
        )
        session.target_diagnostic.connect(lambda payload: diagnostics.append(payload))
        session.finished.connect(lambda message: finished.append(message))
        session.failed.connect(lambda message: failed.append(message))

        click_sent = False
        try:
            session.start("Help me save this.")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not finished and not failed:
                app.processEvents()
                if highlights and not click_sent:
                    x, y, width, height, _label = highlights[0]
                    time.sleep(0.05)
                    session.notify_user_click(x + width // 2, y + height // 2)
                    click_sent = True
                time.sleep(0.01)
            app.processEvents()
        finally:
            if not finished:
                session.cancel()
            thread = session._thread
            if thread is not None:
                thread.join(timeout=1.0)
            session.deleteLater()
            app.processEvents()

        self.assertFalse(failed)
        self.assertEqual(finished, ["Saved."])
        self.assertEqual(highlights, [(40, 50, 120, 32, "Click Save changes.")])
        self.assertGreaterEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["resolution"]["source"], "text_match")
        self.assertEqual(diagnostics[0]["resolution"]["target_id"], "c001")
        self.assertEqual(diagnostics[0]["resolution"]["rect"], (40, 50, 120, 32))


if __name__ == "__main__":
    unittest.main()
