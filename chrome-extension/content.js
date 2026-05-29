const API_BASE_URL = "http://localhost:8000";

let _lastCheckedUrl = null;
let _debounceTimer  = null;
let _blockListener  = null;

function _extractVideoId(url) {
  const match = url.match(/[?&]v=([A-Za-z0-9_-]{11})/);
  return match ? match[1] : null;
}

function _getMode() {
  return new Promise((resolve) => {
    chrome.storage.local.get(['mode'], (res) => resolve(res.mode || 'text'));
  });
}

// Remove the persistent play-block listener from the video element.
function _releaseBlock() {
  if (_blockListener) {
    document.querySelector('video')?.removeEventListener('play', _blockListener);
    _blockListener = null;
  }
}

// Pause via YouTube's native API; fall back to the <video> element.
// Attach a persistent listener that re-pauses on every play attempt.
function _applyBlock() {
  _releaseBlock();

  const ytPlayer = document.getElementById('movie_player');
  if (ytPlayer?.pauseVideo) {
    ytPlayer.pauseVideo();
  } else {
    document.querySelector('video')?.pause();
  }

  const video = document.querySelector('video');
  if (video) {
    _blockListener = () => {
      const p = document.getElementById('movie_player');
      if (p?.pauseVideo) p.pauseVideo(); else video.pause();
    };
    video.addEventListener('play', _blockListener);
  }
}

// Triggered by the "I understand — play anyway" button in overlay.js.
document.addEventListener('guardai:unblock', _releaseBlock);

async function _analyseVideo(url) {
  const videoId = _extractVideoId(url);
  if (!videoId) return;

  _releaseBlock();
  showLoadingOverlay();

  try {
    let result;

    const mode = await _getMode();
    const cacheRes = await fetch(`${API_BASE_URL}/cache/${videoId}?mode=${mode}`);
    if (cacheRes.ok) {
      result = await cacheRes.json();
    } else {
      const analyseRes = await fetch(`${API_BASE_URL}/analyse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, mode }),
      });
      if (!analyseRes.ok) throw new Error(`API returned ${analyseRes.status}`);
      result = await analyseRes.json();
    }

    chrome.runtime.sendMessage({ type: 'RESULT', result });

    if (result.verdict === 'Harmful') {
      chrome.storage.local.get(['blockHarmful'], (res) => {
        if (res.blockHarmful) {
          _applyBlock();
          showBlockedOverlay(result); // full player overlay
        } else {
          showVerdict(result);        // warning banner below player
        }
      });
    } else {
      showVerdict(result);            // safe banner below player
    }
  } catch (_err) {
    showError();
  }
}

function _onUrlChange() {
  const url = window.location.href;
  if (url === _lastCheckedUrl) return;
  if (!url.includes('/watch')) return;

  _lastCheckedUrl = url;
  clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(() => _analyseVideo(url), 1500);
}

window.addEventListener('yt-navigate-finish', _onUrlChange);

const _titleEl = document.querySelector('title');
if (_titleEl) {
  new MutationObserver(_onUrlChange).observe(_titleEl, { childList: true });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'REANALYSE') {
    _releaseBlock();
    _lastCheckedUrl = null;
    // Clear the backend cache so the next analysis uses the current mode
    const videoId = _extractVideoId(window.location.href);
    const invalidate = videoId
      ? fetch(`${API_BASE_URL}/cache/${videoId}`, { method: 'DELETE' }).catch(() => {})
      : Promise.resolve();
    invalidate.then(() => _onUrlChange());
    sendResponse({ ok: true });
  }
});

_onUrlChange();
