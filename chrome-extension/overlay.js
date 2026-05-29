// Overlay rendering — loaded before content.js.

const _SCORE_LABELS = ['ADD', 'SXL', 'PH', 'HH'];
const _SCORE_NAMES  = { ADD: 'Addictive', SXL: 'Sexual / Explicit', PH: 'Physical Harm', HH: 'Hate / Harassment' };

function _esc(str) {
  const d = document.createElement('div');
  d.textContent = str || '';
  return d.innerHTML;
}

function _removeExistingOverlay() {
  document.getElementById('guardai-overlay')?.remove();
  document.getElementById('guardai-player-overlay')?.remove();
}

function _insertBelowPlayer(el) {
  const primary = document.getElementById('primary-inner');
  const player  = document.getElementById('player');
  if (primary)     primary.insertAdjacentElement('afterend', el);
  else if (player) player.insertAdjacentElement('afterend', el);
  else             document.body.prepend(el);
}

function _buildScoreBars(scores) {
  return _SCORE_LABELS.map((label) => {
    const pct  = Math.round(((scores && scores[label]) || 0) * 100);
    const name = _SCORE_NAMES[label] || label;
    return `<div class="guardai-score-row">
      <span class="guardai-score-label" title="${name}">${label}</span>
      <div class="guardai-score-track">
        <div class="guardai-score-fill" style="width:${pct}%"></div>
      </div>
      <span class="guardai-score-value">${pct}%</span>
    </div>`;
  }).join('');
}

function _buildVideoMeta(result) {
  if (!result.title && !result.channel) return '';
  const parts = [];
  if (result.title)   parts.push(`<span class="guardai-meta-title">${_esc(result.title)}</span>`);
  if (result.channel) parts.push(`<span class="guardai-meta-channel">${_esc(result.channel)}</span>`);
  return `<div class="guardai-meta">${parts.join('<span class="guardai-meta-sep">&middot;</span>')}</div>`;
}

// ─── Loading ────────────────────────────────────────────────────────────────

function showLoadingOverlay() {
  _removeExistingOverlay();
  const el = document.createElement('div');
  el.id = 'guardai-overlay';
  el.className = 'guardai-overlay guardai-loading';
  el.innerHTML = `
    <div class="guardai-header">
      <div class="guardai-spinner"></div>
      <span class="guardai-verdict-text">Analysing content&hellip;</span>
    </div>`;
  _insertBelowPlayer(el);
}

// ─── Safe / Harmful verdict (no block) ──────────────────────────────────────

function showVerdict(result) {
  _removeExistingOverlay();
  const isHarmful = result.verdict === 'Harmful';
  const el = document.createElement('div');
  el.id = 'guardai-overlay';
  el.className = `guardai-overlay ${isHarmful ? 'guardai-harmful' : 'guardai-safe'}`;

  const badges = (result.categories || [])
    .map(c => `<span class="guardai-badge">${c}</span>`).join('');

  const verdictHtml = isHarmful
    ? `<span class="guardai-verdict-icon">&#x1F6A8;</span>
       <span class="guardai-verdict-text">Harmful content detected</span>
       ${badges ? `<span class="guardai-badge-group">${badges}</span>` : ''}`
    : `<span class="guardai-verdict-icon">&#x2705;</span>
       <span class="guardai-verdict-text">Safe</span>`;

  const scoreBarsHtml = _buildScoreBars(result.scores);
  const textCues = result.text_cues || [];
  const cuesHtml = textCues.length
    ? `<div class="guardai-cues"><strong>Text cues:</strong> ${
        textCues.map(c => `<span class="guardai-cue-token">${_esc(c.token)}</span>`).join(' ')
      }</div>`
    : '';

  el.innerHTML = `
    <div class="guardai-header">
      <div class="guardai-verdict-row">${verdictHtml}</div>
      <div class="guardai-header-actions">
        <button class="guardai-btn guardai-toggle-btn">Details &#9660;</button>
        <button class="guardai-btn guardai-close-btn" aria-label="Close">&#x2715;</button>
      </div>
    </div>
    ${_buildVideoMeta(result)}
    <div class="guardai-details" hidden>
      <div class="guardai-scores">${scoreBarsHtml}</div>
      ${cuesHtml}
    </div>`;

  el.querySelector('.guardai-close-btn').addEventListener('click', () => el.remove());
  const toggleBtn = el.querySelector('.guardai-toggle-btn');
  const details   = el.querySelector('.guardai-details');
  toggleBtn.addEventListener('click', () => {
    const collapsed = details.hasAttribute('hidden');
    details.toggleAttribute('hidden', !collapsed);
    toggleBtn.innerHTML = collapsed ? 'Details &#9650;' : 'Details &#9660;';
  });

  _insertBelowPlayer(el);
}

// ─── Full-player blocked overlay ─────────────────────────────────────────────

function showBlockedOverlay(result) {
  _removeExistingOverlay();

  const badges = (result.categories || [])
    .map(c => `<span class="guardai-badge guardai-badge-lg">${c}</span>`).join('');

  const el = document.createElement('div');
  el.id = 'guardai-player-overlay';
  el.innerHTML = `
    <div class="guardai-blocked-inner">
      <div class="guardai-blocked-icon">&#x1F6AB;</div>
      <h2 class="guardai-blocked-heading">Video playback is paused</h2>
      <p class="guardai-blocked-reason">due to harmful content for kids</p>
      ${badges ? `<div class="guardai-blocked-badges">${badges}</div>` : ''}
      ${result.title   ? `<p class="guardai-blocked-title">${_esc(result.title)}</p>`   : ''}
      ${result.channel ? `<p class="guardai-blocked-channel">${_esc(result.channel)}</p>` : ''}
      <button class="guardai-unblock-btn">I understand &mdash; play anyway</button>
    </div>`;

  el.querySelector('.guardai-unblock-btn').addEventListener('click', () => {
    document.dispatchEvent(new CustomEvent('guardai:unblock'));
    el.remove();
    showVerdict(result); // show the warning banner after unblocking
  });

  // Inject into the player element so it covers the video exactly
  const player = document.getElementById('movie_player');
  if (player) {
    player.appendChild(el);
  } else {
    // Fallback: position below player
    el.className = 'guardai-overlay guardai-blocked-fallback';
    el.id = 'guardai-overlay';
    _insertBelowPlayer(el);
  }
}

// ─── Error ───────────────────────────────────────────────────────────────────

function showError() {
  _removeExistingOverlay();
  const el = document.createElement('div');
  el.id = 'guardai-overlay';
  el.className = 'guardai-overlay guardai-error';
  el.innerHTML = `
    <div class="guardai-header">
      <span class="guardai-verdict-icon">&#x26A0;&#xFE0F;</span>
      <span class="guardai-verdict-text">Could not analyse this video &mdash; is the backend running?</span>
      <div class="guardai-header-actions">
        <button class="guardai-btn guardai-close-btn" aria-label="Close">&#x2715;</button>
      </div>
    </div>`;
  el.querySelector('.guardai-close-btn').addEventListener('click', () => el.remove());
  _insertBelowPlayer(el);
}

// ─── (removed) showBlockedNote — replaced by showBlockedOverlay ──────────────
