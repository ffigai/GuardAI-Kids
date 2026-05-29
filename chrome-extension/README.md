# GuardAI Kids — Chrome Extension

Analyses YouTube videos for child safety in real time using the GuardAI Kids backend.
Four harm categories are checked: **ADD** (addictive), **SXL** (sexual/explicit), **PH** (physical harm), **HH** (hate/harassment).

---

## Prerequisites

- Python 3.10+ with the `guardaikids` package installed (see repo root)
- A **YouTube Data API v3** key set in the environment: `YOUTUBE_API_KEY=<your-key>`
- Trained model artifacts in the default artifact directory (run `guardaikids train` first)

---

## 1 — Start the backend

From the **repo root**:

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Test it: `curl http://localhost:8000/health` should return `{"status":"ok"}`.

---

## 2 — Load the extension in Chrome

1. Open Chrome and go to `chrome://extensions`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `chrome-extension/` folder from this repository
5. The GuardAI Kids shield icon will appear in the Chrome toolbar

---

## 3 — Usage

Navigate to any `https://www.youtube.com/watch?v=...` URL.  
After ~1.5 s the extension contacts the backend and displays a verdict banner below the player:

- **Green banner** — Safe video, no harm categories triggered
- **Red banner** — Harmful video, with fired category badges (e.g. `[SXL]`, `[PH]`)
- **Amber banner** — Backend unreachable or API error

Click **Details** to expand per-category score bars and text cues.  
Use the popup (toolbar icon) to change the analysis mode or re-analyse the current video.

---

## 4 — Analysis modes

| Mode | What the model uses |
|------|---------------------|
| `text` | Title, description, tags, transcript |
| `image` | Video thumbnail (via CLIP) |
| `multimodal` | Text + thumbnail combined |

The mode is saved to `chrome.storage.local` and persists across sessions.

---

## 5 — Changing the API URL (cloud deployment)

If you deploy the backend to a remote server, update the single constant at the top of [content.js](content.js):

```js
const API_BASE_URL = "https://your-server.example.com";
```

Also update `host_permissions` in [manifest.json](manifest.json) to include the new origin, then reload the extension.

---

## File structure

```
chrome-extension/
├── manifest.json      MV3 extension manifest
├── background.js      Service worker — stores results per tab
├── content.js         Page script — detects navigation, calls API
├── overlay.js         Overlay rendering (loaded before content.js)
├── popup.html         Toolbar popup UI
├── popup.js           Popup logic
├── styles.css         Overlay styles injected into YouTube pages
├── icons/
│   ├── icon16.svg
│   ├── icon48.svg
│   └── icon128.svg
└── README.md          This file
```
