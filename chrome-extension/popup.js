const SCORE_LABELS = ['ADD', 'SXL', 'PH', 'HH'];

function getActiveTab() {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => resolve(tabs[0]));
  });
}

function renderVerdict(data) {
  const banner = document.getElementById('verdict-banner');
  const videoInfo = document.getElementById('video-info');
  const videoTitle = document.getElementById('video-title');
  const videoChannel = document.getElementById('video-channel');
  const scoresEl = document.getElementById('scores-container');
  const timestampEl = document.getElementById('timestamp');

  if (!data || !data.result) {
    banner.className = 'verdict-banner verdict-none';
    banner.textContent = 'No analysis available yet — navigate to a YouTube video.';
    return;
  }

  const { result, timestamp } = data;
  const isHarmful = result.verdict === 'Harmful';

  banner.className = `verdict-banner ${isHarmful ? 'verdict-harmful' : 'verdict-safe'}`;
  const badges = (result.categories || [])
    .map((c) => `<span class="badge">${c}</span>`)
    .join(' ');
  banner.innerHTML = isHarmful
    ? `&#x1F6A8; Harmful${badges ? ' &mdash; ' + badges : ''}`
    : '&#x2705; Safe &mdash; No harm categories detected';

  if (result.title || result.channel) {
    videoInfo.removeAttribute('hidden');
    videoTitle.textContent = result.title || '';
    videoChannel.textContent = result.channel ? `by ${result.channel}` : '';
  }

  if (result.scores) {
    scoresEl.removeAttribute('hidden');
    scoresEl.innerHTML = SCORE_LABELS.map((label) => {
      const pct = Math.round(((result.scores[label]) || 0) * 100);
      return `<div class="score-row">
        <span class="score-label">${label}</span>
        <div class="score-track"><div class="score-fill" style="width:${pct}%"></div></div>
        <span class="score-value">${pct}%</span>
      </div>`;
    }).join('');
  }

  if (timestamp) {
    const d = new Date(timestamp);
    timestampEl.textContent = `Last checked: ${d.toLocaleTimeString()}`;
  }
}

async function init() {
  const tab = await getActiveTab();

  if (!tab?.url?.includes('youtube.com/watch')) {
    const banner = document.getElementById('verdict-banner');
    banner.className = 'verdict-banner verdict-none';
    banner.textContent = 'No YouTube video detected.';
    document.getElementById('reanalyse-btn').disabled = true;
    return;
  }

  // Restore saved preferences
  chrome.storage.local.get(['mode', 'blockHarmful'], (res) => {
    document.getElementById('mode-select').value = res.mode || 'text';
    document.getElementById('block-toggle').checked = !!res.blockHarmful;
  });

  // Fetch latest result from background service worker
  chrome.runtime.sendMessage({ type: 'GET_RESULT' }, (data) => {
    renderVerdict(data);
  });

  // Persist setting changes
  document.getElementById('mode-select').addEventListener('change', (e) => {
    chrome.storage.local.set({ mode: e.target.value });
  });
  document.getElementById('block-toggle').addEventListener('change', (e) => {
    chrome.storage.local.set({ blockHarmful: e.target.checked });
  });

  // Re-analyse: tell content script to reset and re-run
  document.getElementById('reanalyse-btn').addEventListener('click', () => {
    chrome.tabs.sendMessage(tab.id, { type: 'REANALYSE' }, () => window.close());
  });
}

init();
