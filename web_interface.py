"""Basic web interface for submitting a YouTube URL and reading recommendations."""

from __future__ import annotations

import os

from flask import Flask, render_template_string, request

from etp.service import analyze_youtube_url

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GuardAI Kids: Age-Aware YouTube Content Safety Analyzer</title>
  <style>
    :root {
      --bg: #f7f1e6;
      --panel: #fffaf2;
      --ink: #1f2933;
      --muted: #52606d;
      --accent: #d9822b;
      --line: #e6d5be;
      --allow: #2f855a;
      --warn: #c05621;
      --block: #c53030;
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff3dc 0, transparent 28%),
        linear-gradient(180deg, #f6efe3 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 920px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }
    .hero, .panel, .result {
      background: color-mix(in srgb, var(--panel) 92%, white 8%);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
    }
    .hero {
      padding: 28px;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0;
      font-size: clamp(2.8rem, 7vw, 5rem);
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
      font-size: clamp(0.95rem, 2vw, 1.35rem);
      line-height: 1.3;
      letter-spacing: 0.02em;
      color: var(--muted);
      max-width: 34rem;
    }
    .hero p, .meta, .error {
      color: var(--muted);
    }
    .panel {
      padding: 20px;
      margin-bottom: 20px;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }
    input[type=text] {
      width: 100%;
      padding: 14px 16px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font-size: 1rem;
      box-sizing: border-box;
    }
    button {
      margin-top: 14px;
      background: var(--accent);
      color: white;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font-size: 1rem;
      cursor: pointer;
    }
    .result {
      padding: 20px;
      margin-top: 18px;
    }
    .cards {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-top: 18px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.75);
    }
    .decision {
      font-weight: 700;
      margin-bottom: 10px;
    }
    .Allow { color: var(--allow); }
    .Warn { color: var(--warn); }
    .Block { color: var(--block); }
    .scores {
      margin-top: 18px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      font-size: 0.95rem;
    }
    .error {
      margin-top: 12px;
      font-weight: 700;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-title">
        <h1>GuardAI Kids</h1>
        <small>Age-Aware YouTube Content Safety Analyzer</small>
      </div>
      <p>Paste a YouTube link to assess whether the video should be allowed, warned, or blocked for ages 0-4, 5-8, and 9-12.</p>
    </section>

    <section class="panel">
      <form method="post">
        <label for="url">YouTube URL</label>
        <input id="url" name="url" type="text" value="{{ url }}" placeholder="https://www.youtube.com/watch?v=..." required>
        <button type="submit">Analyze Video</button>
      </form>
      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
    </section>

    {% if result %}
      <section class="result">
        <h2>{{ result.metadata.title }}</h2>
        <div class="meta">{{ result.metadata.channel }} | {{ result.metadata.published_at }}</div>
        <div class="cards">
          {% for age_group, rec in result.recommendations.items() %}
            <article class="card">
              <div><strong>Ages {{ age_group.replace('_', '-') }}</strong></div>
              <div class="decision {{ rec.decision }}">{{ rec.decision }}</div>
              <div>{{ rec.explanation }}</div>
            </article>
          {% endfor %}
        </div>
        <div class="scores">
          <strong>Model Scores:</strong>
          {% for label, score in result.model_scores.items() %}
            <span>{{ label }}={{ '%.3f'|format(score) }}</span>{% if not loop.last %} | {% endif %}
          {% endfor %}
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

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        try:
            result = analyze_youtube_url(url, api_key)
        except Exception as exc:
            error = str(exc)

    return render_template_string(PAGE_TEMPLATE, result=result, error=error, url=url)


if __name__ == "__main__":
    app.run(debug=True)
