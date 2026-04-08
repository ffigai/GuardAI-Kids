"""Basic web interface for submitting a YouTube URL and reading recommendations."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template_string, request

from guardaikids.config import CATEGORY_DESCRIPTIONS, LABELS_ORDER, MODE
from guardaikids.service import analyze_youtube_url

app = Flask(__name__)

CATEGORY_COLORS = {
    "ADD": "#ff8a00",
    "SXL": "#ff2b2b",
    "PH": "#111111",
    "HH": "#a020f0",
}

VIEW_MODES = ("basic", "detailed")

# display_name -> (mode, artifact_subdir, image_analysis_model)
MODEL_OPTIONS = {
    "text":       ("text",       "text",                       "clip"),
    "image":      ("image",      "image_clip_nsfw_violence",   "clip_nsfw_violence"),
    "multimodal": ("multimodal", "multimodal_clip_nsfw_violence", "clip_nsfw_violence"),
}
DEFAULT_MODEL_OPTION = "multimodal"
REPO_ROOT = Path(__file__).resolve().parents[2]

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GuardAI Kids: YouTube Content Safety Analyzer</title>
  <style>
    :root {
      --bg: #f7f7f8;
      --panel: #ffffff;
      --ink: #0f0f0f;
      --muted: #606060;
      --accent: #ff0000;
      --line: #dadce0;
      --safe: #16a34a;
      --harmful: #dc2626;
      --add: #ff8a00;
      --sxl: #ff2b2b;
      --ph: #111111;
      --hh: #a020f0;
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(255, 0, 0, 0.08) 0, transparent 24%),
        linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      width: min(96vw, 1560px);
      max-width: 1560px;
      margin: 0 auto;
      padding: 18px 20px 36px;
      box-sizing: border-box;
    }
    .hero, .panel, .result {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 24px rgba(15, 15, 15, 0.06);
    }
    .top-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      align-items: stretch;
      margin-bottom: 14px;
    }
    .hero {
      padding: 22px 26px;
      height: 100%;
      box-sizing: border-box;
    }
    h1 {
      margin: 0;
      font-size: clamp(2.2rem, 5vw, 3.7rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero-title {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 12px;
    }
    .hero-title small {
      display: block;
      font-size: clamp(0.88rem, 1.8vw, 1.08rem);
      line-height: 1.3;
      letter-spacing: 0.02em;
      color: var(--muted);
      max-width: 34rem;
    }
    .hero p, .meta, .error, .legend-copy {
      color: var(--muted);
    }
    .hero p {
      font-size: 0.92rem;
      line-height: 1.42;
      margin-bottom: 0;
    }
    .panel {
      padding: 16px 18px;
      height: 100%;
      box-sizing: border-box;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 0.9rem;
    }
    .form-row {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .form-row label {
      margin-bottom: 0;
    }
    input[type=text] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font-size: 0.9rem;
      box-sizing: border-box;
    }
    select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font-size: 0.9rem;
      box-sizing: border-box;
      background: white;
      margin-bottom: 12px;
    }
    button {
      margin-top: 14px;
      background: var(--accent);
      color: white;
      border: 0;
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 0.88rem;
      cursor: pointer;
    }
    .result {
      padding: 20px;
      margin-top: 12px;
    }
    .verdict-banner {
      display: flex;
      align-items: center;
      gap: 18px;
      margin-top: 20px;
      padding: 18px 22px;
      border-radius: 16px;
      border: 2px solid var(--line);
    }
    .verdict-banner.Harmful { border-color: var(--harmful); background: rgba(220,38,38,0.05); }
    .verdict-banner.Safe    { border-color: var(--safe);    background: rgba(22,163,74,0.05); }
    .verdict-label {
      font-size: 2.6rem;
      font-weight: 700;
      line-height: 1;
    }
    .verdict-label.Harmful { color: var(--harmful); }
    .verdict-label.Safe    { color: var(--safe); }
    .category-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .category-tag {
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 700;
      color: white;
    }
    .category-tag.ADD { background: var(--add); }
    .category-tag.SXL { background: var(--sxl); }
    .category-tag.PH  { background: var(--ph); color: white; }
    .category-tag.HH  { background: var(--hh); }
    .decision {
      font-weight: 700;
      margin-bottom: 10px;
    }
    .Safe    { color: var(--safe); }
    .Harmful { color: var(--harmful); }
    .error {
      margin-top: 12px;
      font-weight: 700;
    }
    .thumb {
      margin-top: 0;
    }
    .thumb img {
      width: 100%;
      display: block;
      border-radius: 16px;
      border: 1px solid var(--line);
      box-shadow: 0 8px 20px rgba(15, 15, 15, 0.12);
    }
    .mode-chip {
      display: inline-block;
      margin-top: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255, 0, 0, 0.1);
      color: var(--accent);
      font-weight: 700;
      text-transform: capitalize;
    }
    .topline {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .view-toggle {
      display: inline-flex;
      gap: 8px;
      padding: 4px;
      border-radius: 999px;
      background: rgba(255, 0, 0, 0.08);
      flex: 1 1 320px;
    }
    .view-toggle button {
      margin-top: 0;
      background: transparent;
      color: var(--accent);
      border: 0;
      padding: 7px 9px;
      flex: 1 1 0;
      font-size: 0.84rem;
    }
    .view-toggle button.active {
      background: var(--accent);
      color: white;
    }
    .action-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
    }
    .action-row > button {
      margin-top: 0;
      flex: 1 1 220px;
      min-height: 38px;
    }
    .header-block {
      display: grid;
      gap: 16px;
      grid-template-columns: minmax(220px, 280px) 1fr;
      align-items: start;
      margin-top: 16px;
    }
    .header-block.text-only {
      grid-template-columns: 1fr;
    }
    .decision-legend {
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff9f9;
    }
    .decision-legend-grid {
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr;
      margin-top: 10px;
    }
    .decision-pill {
      font-weight: 700;
      font-size: 0.86rem;
    }
    .legend {
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fafafa;
    }
    .footer-panels {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 18px;
      align-items: stretch;
    }
    .footer-panels .legend,
    .footer-panels .decision-legend {
      margin-top: 0;
      height: 100%;
      box-sizing: border-box;
    }
    .legend-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr;
      margin-top: 10px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.82rem;
    }
    .legend.basic {
      font-size: 0.8rem;
    }
    .legend.basic .legend-grid {
      gap: 8px;
    }
    .legend.basic .legend-item {
      font-size: 0.8rem;
    }
    .footer-panels strong {
      font-size: 0.94rem;
    }
    .footer-panels .meta {
      font-size: 0.8rem;
      line-height: 1.3;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      flex: 0 0 14px;
    }
    .swatch.ADD { background: var(--add); }
    .swatch.SXL { background: var(--sxl); }
    .swatch.PH { background: var(--ph); }
    .swatch.HH { background: var(--hh); }
    .chart {
      margin-top: 14px;
      display: grid;
      gap: 12px;
    }
    .row {
      display: grid;
      grid-template-columns: 48px 1fr 44px;
      gap: 12px;
      align-items: center;
      font-size: 0.92rem;
    }
    .row.primary {
      transform: scale(1.02);
      font-weight: 700;
    }
    .label {
      font-weight: 700;
    }
    .track {
      position: relative;
      height: 16px;
      border-radius: 999px;
      background: #ececec;
      overflow: visible;
    }
    .fill {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      border-radius: 999px;
      background: var(--accent);
    }
    .fill.ADD { background: var(--add); }
    .fill.SXL { background: var(--sxl); }
    .fill.PH { background: var(--ph); }
    .fill.HH { background: var(--hh); }
    .marker {
      position: absolute;
      top: -4px;
      bottom: -4px;
      width: 2px;
      background: rgba(31, 41, 51, 0.45);
    }
    .marker.threshold {
      background: rgba(197, 48, 48, 0.85);
    }
    .score {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .cue-box {
      margin-top: 16px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fafafa;
    }
    .explain {
      margin: 14px 0 0;
      padding-left: 20px;
    }
    .explain li {
      margin-bottom: 6px;
    }
    .cue-title {
      font-weight: 700;
      margin-bottom: 10px;
    }
    .cue-columns {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr 1fr;
    }
    .cue-list {
      margin: 0;
      padding-left: 18px;
    }
    .cue-list li {
      margin-bottom: 6px;
    }
    .summary-box {
      margin-top: 18px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      font-size: 0.95rem;
    }
    .description-box {
      margin-top: 16px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fafafa;
    }
    .description-box p {
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.5;
      white-space: pre-wrap;
    }
    @media (max-width: 720px) {
      .wrap {
        width: 100%;
        max-width: none;
        padding: 16px 14px 26px;
      }
      .top-grid,
      .footer-panels,
      .header-block,
      .thumb,
      .cue-columns {
        grid-template-columns: 1fr;
      }
      .form-row {
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .row {
        grid-template-columns: 42px 1fr 42px;
      }
    }
    @media (min-width: 721px) and (max-width: 1180px) {
      .top-grid {
        grid-template-columns: 1fr;
      }
      .footer-panels {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top-grid">
      <section class="hero">
        <div class="hero-title">
          <h1>GuardAI Kids</h1>
          <small>YouTube Content Safety Analyzer for Children</small>
        </div>
        <p>Paste a YouTube link to assess whether the video is safe or harmful for children, with category-level explanations.</p>
      </section>

      <section class="panel">
        <form method="post">
          <div class="form-row">
            <label for="url">YouTube URL:</label>
            <input id="url" name="url" type="text" value="{{ url }}" placeholder="https://www.youtube.com/watch?v=..." required>
          </div>
          <div class="form-row">
            <label for="model_option">Model:</label>
            <select id="model_option" name="model_option">
              {% for option in model_options %}
                <option value="{{ option }}" {% if option == selected_model_option %}selected{% endif %}>{{ option }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="action-row">
            <button type="submit">Analyze Content</button>
            <div class="view-toggle">
              <button type="submit" name="view" value="basic" class="{% if selected_view == 'basic' %}active{% endif %}">Basic View</button>
              <button type="submit" name="view" value="detailed" class="{% if selected_view == 'detailed' %}active{% endif %}">Detailed View</button>
            </div>
          </div>
        </form>
        {% if error %}
          <div class="error">{{ error }}</div>
        {% endif %}
      </section>
    </div>

    {% if result %}
      <section class="result">
        <div class="header-block {% if result.analysis_mode == 'text' or not result.metadata.thumbnail_url %}text-only{% endif %}">
          {% if result.analysis_mode != 'text' and result.metadata.thumbnail_url %}
            <div class="thumb">
              <img src="{{ result.metadata.thumbnail_url }}" alt="YouTube thumbnail for {{ result.metadata.title }}">
            </div>
          {% endif %}
          <div>
            <h2>{{ result.metadata.title }}</h2>
            <div class="meta">{{ result.metadata.channel }} | {{ result.metadata.published_at }}</div>
            <div class="mode-chip">{{ result.analysis_mode }} mode</div>
          </div>
        </div>

        <div class="verdict-banner {{ result.decision }}">
          <div class="verdict-label {{ result.decision }}">{{ result.decision }}</div>
          <div class="category-tags">
            {% for cat in result.categories %}
              <span class="category-tag {{ cat }}">{{ cat }}</span>
            {% endfor %}
            {% if not result.categories %}
              <span style="color: var(--safe); font-weight:700;">No harm categories detected</span>
            {% endif %}
          </div>
        </div>

        <div class="chart" style="margin-top:20px;">
          {% for label in labels_order %}
            {% set score = result.get('model_scores', {}).get(label, 0.0) %}
            {% set threshold = result.training_metadata.get('f2_thresholds', {}).get(label, 0.3) %}
            <div class="row {% if result.trigger_category == label %}primary{% endif %}">
              <div class="label">{{ label }}</div>
              <div class="track">
                <div class="fill {{ label }}" style="width: {{ '%.2f'|format(score * 100) }}%; {% if result.trigger_category == label %}height: 20px; top: -2px;{% endif %}"></div>
                <div class="marker threshold" style="left: {{ '%.2f'|format(threshold * 100) }}%"></div>
              </div>
              <div class="score">{{ '%.0f'|format(score * 100) }}%</div>
            </div>
          {% endfor %}
        </div>

        {% if selected_view == 'detailed' %}
          <ul class="explain">
            {% for bullet in result.explanation_bullets %}
              <li>{{ bullet }}</li>
            {% endfor %}
          </ul>
        {% endif %}

        <div class="cue-box" style="margin-top:18px;">
          <div class="cue-title">Evidence Signals</div>
          <div class="cue-columns">
            <div>
              <strong>Text cues</strong>
              <ul class="cue-list">
                {% if result.top_tokens %}
                  {% for token in result.top_tokens[:4] %}
                    <li>{{ token.token }}</li>
                  {% endfor %}
                {% else %}
                  <li>No dominant text cue surfaced.</li>
                {% endif %}
              </ul>
            </div>
            <div>
              <strong>Image cues</strong>
              <ul class="cue-list">
                {% if result.image_highlights %}
                  {% for cue in result.image_highlights[:4] %}
                    <li>{{ cue.label }}{% if cue.score is defined %} ({{ '%.0f'|format(cue.score * 100) }}%){% endif %}</li>
                  {% endfor %}
                {% else %}
                  <li>No dominant thumbnail cue surfaced.</li>
                {% endif %}
              </ul>
            </div>
          </div>
        </div>

        {% if selected_view == 'detailed' %}
          {% if result.metadata.description %}
            <section class="description-box">
              <strong>Video Description</strong>
              <p>{{ result.metadata.description }}</p>
            </section>
          {% endif %}
          <div class="summary-box">
            <strong>Model Scores:</strong>
            {% for label in labels_order %}
              <span>{{ label }}={{ '%.0f'|format(result.get('model_scores', {}).get(label, 0.0) * 100) }}%</span>{% if not loop.last %} | {% endif %}
            {% endfor %}
          </div>
        {% endif %}

        <div class="footer-panels">
          {% if selected_view == 'basic' %}
            <section class="legend basic">
              <strong>Category Legend</strong>
              <div class="legend-grid">
                {% for label in labels_order %}
                  <div class="legend-item">
                    <span class="swatch {{ label }}"></span>
                    <span><strong>{{ label }}</strong>: {{ category_descriptions[label] }}</span>
                  </div>
                {% endfor %}
              </div>
            </section>
          {% else %}
            <section class="legend basic">
              <strong>Category Legend</strong>
              <div class="legend-copy">Abbreviations used in the charts and trigger labels.</div>
              <div class="legend-grid">
                {% for label in labels_order %}
                  <div class="legend-item">
                    <span class="swatch {{ label }}"></span>
                    <span><strong>{{ label }}</strong>: {{ category_descriptions[label] }}</span>
                  </div>
                {% endfor %}
              </div>
            </section>
          {% endif %}

          <section class="decision-legend">
            <strong>Decision Guide</strong>
            <div class="decision-legend-grid">
              <div>
                <span class="decision-pill Safe">SAFE</span>
                <span class="meta"> - No harm category exceeded the detection threshold. Content appears appropriate for children.</span>
              </div>
              <div>
                <span class="decision-pill Harmful">HARMFUL</span>
                <span class="meta"> - One or more harm categories exceeded the threshold. Category tags indicate what was detected.</span>
              </div>
            </div>
          </section>
        </div>
      </section>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    url = ""
    selected_model_option = DEFAULT_MODEL_OPTION
    selected_view = "basic"

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        selected_model_option = request.form.get("model_option", DEFAULT_MODEL_OPTION).strip() or DEFAULT_MODEL_OPTION
        if selected_model_option not in MODEL_OPTIONS:
            selected_model_option = DEFAULT_MODEL_OPTION
        selected_view = request.form.get("view", "basic").strip() or "basic"
        if selected_view not in VIEW_MODES:
            selected_view = "basic"

        selected_mode, artifact_subdir, image_analysis_model = MODEL_OPTIONS[selected_model_option]
        artifact_dir = REPO_ROOT / "artifacts" / artifact_subdir
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        try:
            result = analyze_youtube_url(
                url,
                api_key,
                mode=selected_mode,
                artifact_dir=artifact_dir,
                image_analysis_model=image_analysis_model,
            )
        except Exception as exc:
            error = str(exc)
    else:
        selected_mode, _, _ = MODEL_OPTIONS[selected_model_option]

    return render_template_string(
        PAGE_TEMPLATE,
        result=result,
        error=error,
        url=url,
        selected_model_option=selected_model_option,
        selected_view=selected_view,
        model_options=list(MODEL_OPTIONS.keys()),
        labels_order=LABELS_ORDER,
        category_descriptions=CATEGORY_DESCRIPTIONS,
        category_colors=CATEGORY_COLORS,
    )


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
