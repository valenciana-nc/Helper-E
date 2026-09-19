<div align="center">
  <img src="assets/helper_logo.png" alt="Orb Helper logo" width="180" />
  <h1>HELPER</h1>
  <p><strong>A floating Windows assistant with chat, voice, screen awareness, and guided computer help.</strong></p>
</div>

## What We Are Building

Orb Helper is a desktop assistant that lives as a small floating orb on Windows. It can open a chat window, listen for voice input, understand the current screen, guide the user with overlays, and optionally control the mouse and keyboard after confirmation.

The goal is a helper that feels always available but stays out of the way: quick to summon, clear about what it is doing, and careful around risky actions.

## Core Pieces

- **Floating orb:** a PyQt6 widget that stays on top and acts as the main entry point.
- **Dashboard:** account, voice, model, and behavior settings.
- **Chat:** typed conversations backed by ChatGPT sign-in.
- **Voice:** transcription and spoken replies through `OPENAI_API_KEY`.
- **Screen capture:** lets the assistant reason about what is visible.
- **Help mode:** shows ghost cursor guidance and highlight rectangles while the user clicks.
- **Active mode:** can execute actions, with confirmation for risky steps.

## Setup

```powershell
cd $env:USERPROFILE\helperEngine
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Run the orb:

```powershell
.\.venv\Scripts\pythonw.exe main.py
```

Run with a visible console while debugging:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Sign-In

Chat and computer-use features use the **Sign in with ChatGPT** button in the dashboard. The app still starts when signed out, but chat and computer-use stay disabled until sign-in succeeds.

Voice transcription and spoken replies are separate and need `OPENAI_API_KEY` in `.env`.

## Configuration

Canonical settings use the `HELPER_*` prefix. Legacy `HELPLER_*` and `HARVIS_*` keys are still read as fallbacks, but the dashboard saves new changes as `HELPER_*`.

Useful settings:

- `HELPER_HOTKEY=ctrl+shift+space`
- `HELPER_DEFAULT_MODE=help`
- `HELPER_AGENT_MODEL=gpt-5.5`
- `HELPER_REASONING_MODEL=gpt-5.5`
- `HELPER_SPEAK_TYPED_CHAT=false`
- `OPENAI_API_KEY=` for voice only

## Modes

- **Help mode:** walks through a task with a ghost cursor and persistent highlight rectangles while the user does the clicks.
- **Active mode:** can execute actions. Risky actions, destructive text, purchases, sends, and terminal Enter require confirmation.


## Privacy and Safety

- Local `.env` files, logs, screenshots, app data, virtual environments, and assistant state are ignored by git.
- ChatGPT sign-in tokens are stored in the OS keyring when possible.
- Active mode is opt-in for each session and keeps confirmation prompts around risky actions.

## Troubleshooting

- If sign-in fails, close other Helper instances and retry. OAuth needs local callback port `127.0.0.1:1455`.
- If chat says you are not signed in, open Dashboard > Account and sign in with ChatGPT.
- If voice says an API key is missing, add `OPENAI_API_KEY` in Dashboard > Account.
- Logs live in `logs/startup.log`.
- If the shortcut stops working after a folder rename, rerun `scratch/create_shortcut.ps1`.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m pip check
```

Help-mode target precision checks:

```powershell
.\.venv\Scripts\python.exe -m help_highlight_qa --artifacts logs/help_qa/latest
.\.venv\Scripts\python.exe -m help_live_probe --capture primary --artifacts logs/help_live_probe/primary
.\.venv\Scripts\python.exe -m help_precision_selftest --artifacts logs/help_precision_selftest/latest
```

`help_highlight_qa` runs model-free synthetic scenarios and writes failure
artifacts when a target resolves incorrectly. `help_live_probe` captures the
desktop, collects Windows UI Automation candidates, and draws their rectangles
over the screenshot so monitor/DPI alignment can be inspected.
`help_precision_selftest` opens a known local test window, captures it, resolves
the Save button through the same Help targeting pipeline, and writes pass/fail
artifacts.

## Development checks

From the project root on Windows:

```powershell
python -m ruff check .
python -m compileall -q -x ".*\.venv.*" .
python -m unittest discover -s tests -p "test*.py"
```

Pull requests run the same checks on Windows through GitHub Actions.
