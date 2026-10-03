# GENEVIEVE Super Response Engine

**Canonical Python engine repository.**

Repository family:
- `Tracey-s-Super-Computer` — canonical Python engine
- `Traceys-super-computer` — exact historical duplicate with the same five file blobs, retained as reference only
- later web-based Super Response builds are a separate implementation track and are not merged into this Python engine

A safer, current replacement for the original multi-model script.

It sends one question to selected providers in parallel, records which calls
succeeded, and asks an available provider to synthesize only the successful
answers.

## Improvements

- Uses the current OpenAI Responses API.
- Uses Anthropic's async Messages API.
- Uses Google's current `google-genai` SDK rather than the legacy
  `google-generativeai` package.
- Runs providers concurrently.
- Adds timeouts, limited retries, latency and token reporting.
- Skips providers whose keys are missing.
- Keeps provider failures separate from actual answers.
- Treats candidate answers as untrusted data during synthesis, reducing prompt
  injection risk.
- Falls back to another synthesizer when the preferred one fails.
- Supports Markdown or JSON export.
- Does not store OpenAI Responses API output (`store=False`).
- Reads keys from `.env`; keys are excluded by `.gitignore`.

## Important privacy and cost warning

A question is sent to every selected provider. Successful answers are then sent
to the synthesis provider. This means the question and candidate answers cross
multiple external services and can create several billable API calls.

Do not submit confidential client records, health information, passwords,
private legal material, API keys, or protected business information unless you
have established appropriate consent, contracts, provider settings, and data
handling controls.

Multi-model agreement is not proof that an answer is correct.

## Windows setup

Open PowerShell in this folder.

### 1. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When PowerShell blocks activation, run this once in the same window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
py -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create your private environment file

```powershell
Copy-Item .env.example .env
notepad .env
```

Add only the keys for providers you intend to use.

### 4. Run interactively

```powershell
py super_response.py
```

Or provide the question directly:

```powershell
py super_response.py "Explain the strongest and weakest parts of this proposal."
```

## Useful commands

Choose providers:

```powershell
py super_response.py --providers openai,claude "Your question"
```

Use Gemini as the first-choice synthesizer:

```powershell
py super_response.py --synthesizer gemini "Your question"
```

Show individual answers:

```powershell
py super_response.py --show-raw "Your question"
```

Save a Markdown report:

```powershell
py super_response.py --show-raw --save output\answer.md "Your question"
```

Save JSON:

```powershell
py super_response.py --json --save output\answer.json "Your question"
```

Skip the extra synthesis call:

```powershell
py super_response.py --no-synthesis --show-raw "Your question"
```

## Model controls

The default model names are deliberately configurable because providers change
their model catalogues over time.

Edit `.env`:

```dotenv
ANTHROPIC_MODEL=claude-sonnet-5
OPENAI_MODEL=gpt-5.6-terra
GEMINI_MODEL=gemini-3.6-flash
SYNTHESIZER_PROVIDER=auto
```

A provider model may not be enabled on every account. Replace a default with a
model available to your account when necessary.

## Files

- `super_response.py` — application
- `.env.example` — safe configuration template
- `requirements.txt` — Python dependencies
- `.gitignore` — prevents secrets and local outputs entering Git
- `README.md` — setup and usage
- `tests/test_core.py` — offline core-behaviour tests

## GitHub safety

Before committing:

```powershell
git status
```

Confirm `.env` is not listed. Never commit an API key. If a key is accidentally
committed, revoke and replace it immediately; deleting the file from the newest
commit does not remove the key from Git history.
## Run the offline tests

These tests do not call any AI provider or spend API credit.

```powershell
py -m unittest discover -s tests -v
```
