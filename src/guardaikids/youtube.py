"""YouTube metadata and inference helpers."""

import re


def _import_youtube_dependencies():
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "YouTube support requires the optional dependency 'google-api-python-client'."
        ) from exc

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise ImportError(
            "YouTube support requires the optional dependency 'youtube-transcript-api'."
        ) from exc

    return build, YouTubeTranscriptApi


def build_youtube_client(api_key: str):
    build, _ = _import_youtube_dependencies()
    return build("youtube", "v3", developerKey=api_key)


def extract_video_id(url: str) -> str | None:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_youtube_metadata(url: str, youtube_client) -> dict[str, str] | None:
    _, transcript_api = _import_youtube_dependencies()
    video_id = extract_video_id(url)
    if not video_id:
        return None

    response = youtube_client.videos().list(part="snippet", id=video_id).execute()
    if not response.get("items"):
        return None

    snippet = response["items"][0]["snippet"]
    thumbnails = snippet.get("thumbnails", {})
    transcript_text = ""
    try:
        transcript = transcript_api.get_transcript(video_id)
        transcript_text = " ".join(item["text"] for item in transcript)
    except Exception:
        transcript_text = ""

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "thumbnail_url": (
            thumbnails.get("maxres", {}).get("url")
            or thumbnails.get("standard", {}).get("url")
            or thumbnails.get("high", {}).get("url")
            or thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url", "")
        ),
        "transcript": transcript_text,
    }


def build_model_input(metadata: dict[str, str]) -> str:
    return f"{metadata['title']} {metadata['description']} {metadata['transcript']}".strip()
