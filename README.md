<<<<<<< HEAD
# browser-automation-tool
=======
# VLM GitHub Releases Navigator

A CLI tool that uses a vision-language model (VLM) plus Playwright to navigate GitHub without any CSS selectors or XPath. The agent makes all navigation decisions from screenshots and executes only bounding‑box actions (click/type/scroll/back), then extracts the latest release info as structured JSON.

It is designed to be fully selector‑free and reproducible from a fresh clone, with debug traces saved to disk for inspection.

## Demo flow (GitHub)
1) Open github.com
2) Search for the target repo (default: openclaw/openclaw)
3) Click into the repository
4) Open Releases
5) Extract latest release fields

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

### Configuration
Set required environment variables:
- `OPENAI_API_KEY` (required)

Optional model overrides:
- `OPENAI_MODEL_VLM` (default: gpt-5-mini)
- `OPENAI_MODEL_ROUTER` (default: gpt-5-nano)
- `OPENAI_MODEL_PROMPT` (default: gpt-5-mini)

Two ways to configure:

Option A: export directly
```bash
export OPENAI_API_KEY="your_key_here"
```

Option B: use a .env file
```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Usage
Default repo:
```bash
python navigate.py --repo "openclaw/openclaw"
```

Custom start URL and prompt:
```bash
python navigate.py --url "https://github.com" --prompt "search for openclaw and get the current release and related tags"
```

Headed mode (debug):
```bash
python navigate.py --repo "openclaw/openclaw" --headed
```

Override models:
```bash
python navigate.py --repo "openclaw/openclaw" --vlm-model gpt-5-mini --router-model gpt-5-nano --prompt-model gpt-5-mini
```

Write JSON to file:
```bash
python navigate.py --repo "openclaw/openclaw" --out output.json
```

## Output
The tool prints JSON to stdout:
```json
{
  "repository": "openclaw/openclaw",
  "latest_release": {
    "version": "...",
    "tag": "...",
    "author": "..."
  }
}
```

## How the vision agent works (high level)
- **Observe**: capture a raw Playwright screenshot (PNG bytes), URL, and title.
- **Decide**: send the raw screenshot + subgoal to the VLM; receive a strict JSON action.
- **Act**: execute the action via Playwright mouse/keyboard primitives only.
- **Validate**: check URL/screenshot hashes for progress, and recover on stalls.
- **Extract**: once on Releases, use the VLM to parse the latest release fields.

### Region exploration fallback
On GitHub search results, the model returns a **coarse bounding box** for the first result card. The agent then performs deterministic “region exploration” within that box (grid clicks, multi‑round shrinking toward the top‑left) until the URL changes to the target repo. This avoids tight‑bbox brittleness without using selectors.

## Debug artifacts
Each run saves artifacts to `runs/<timestamp>/`:
- `step_01.png`, `step_02.png`, ...
- `step_01_observation.json`
- `step_01_action.json`

Annotated images (bbox overlays) are for **debug only** and are never used as model input.

## Limitations & failure modes
- UI changes, cookie banners, or promotions can degrade navigation accuracy.
- GitHub rate limits or bot detection can block navigation.
- Small clickable targets (icons/badges) are harder for pure vision to hit reliably.
- The agent does not read DOM text; all extraction is from screenshots.

## Safety & privacy
- **No login state is stored or shipped.**
- Browser profiles, cookies, and caches are ignored via `.gitignore` and are not part of the repo.
- Only raw screenshots are sent to the VLM; annotated debug images are never used for inference.

## Reproducibility notes
- Default viewport: 1280x720
- Default limits: max steps = 25, max retries per stage = 3
- Run headed with `--headed` to visually debug.
- If you need slower interactions, adjust `slow_mo_ms` or `action_delay_ms` in `tools/browser.py`.
>>>>>>> 2125778 (Initial commit)
