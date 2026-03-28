#!/usr/bin/env python3
"""Web interface for Audio Quote Finder.

Run with:
    python web_app.py
    python web_app.py --port 8080
    python web_app.py --audio-dir /path/to/mp3s

Then open http://localhost:5000 in your browser.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Reuse the core logic from find_quote.py
from find_quote import (
    DEFAULT_STORE,
    find_quote,
    fmt_time,
    list_cached,
    _get_context,
)

app = Flask(__name__)
CORS(app, origins=["https://drcaley.github.io"])

BOOKS_DIR = Path(__file__).parent / "books"
CLIPS_DIR = Path(__file__).parent / "clips"
CLIPS_DIR.mkdir(exist_ok=True)
CLIP_STATS_FILE = CLIPS_DIR / "_stats.json"


def _load_clip_stats() -> dict:
    """Load clip popularity stats."""
    if CLIP_STATS_FILE.exists():
        with open(CLIP_STATS_FILE) as f:
            return json.load(f)
    return {}


def _save_clip_stats(stats: dict):
    """Save clip popularity stats."""
    with open(CLIP_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def _track_clip(clip_name: str, quote: str, audio_name: str, start: float, end: float):
    """Increment the play count for a clip."""
    stats = _load_clip_stats()
    if clip_name in stats:
        stats[clip_name]["count"] += 1
    else:
        stats[clip_name] = {
            "count": 1,
            "quote": quote,
            "audio_name": audio_name,
            "start": start,
            "end": end,
        }
    _save_clip_stats(stats)


def _resolve_audio(audio_name: str) -> Path | None:
    """Safely resolve an audio filename to a path within BOOKS_DIR only.

    Prevents path traversal — rejects anything with slashes or '..'.
    """
    # Strip any directory components — only bare filenames allowed
    safe_name = Path(audio_name).name
    if safe_name != audio_name or '..' in audio_name:
        return None
    # Search recursively within BOOKS_DIR
    for f in BOOKS_DIR.rglob(safe_name):
        resolved = f.resolve()
        if resolved.is_relative_to(BOOKS_DIR.resolve()):
            return resolved
    return None


def _load_all_transcripts(store: Path) -> list[dict]:
    """Load all cached transcripts."""
    transcripts = []
    if not store.exists():
        return transcripts
    for tf in sorted(store.rglob("*.json")):
        with open(tf) as f:
            transcripts.append(json.load(f))
    return transcripts


def _search_all(quote: str, threshold: float, store: Path) -> list[dict]:
    """Search all transcripts for a quote."""
    all_matches = []
    for data in _load_all_transcripts(store):
        audio_file = data.get("audio_file", "?")
        words = data.get("words", [])
        matches = find_quote(words, quote, threshold)
        for m in matches:
            m["audio_file"] = audio_file
            m["audio_name"] = Path(audio_file).name
            m["context"] = _get_context(words, m["start_time"], m["end_time"])
        all_matches.extend(matches)

    all_matches.sort(key=lambda m: m["score"], reverse=True)
    return all_matches


def _make_clip(audio_file: str, start: float, end: float, buffer: float = 0.3) -> Path | None:
    """Extract a clip and return the path to the file. Max 30 seconds."""
    MAX_CLIP_SECONDS = 30
    if (end - start) > MAX_CLIP_SECONDS:
        end = start + MAX_CLIP_SECONDS

    ss = max(0, start - buffer)
    to = end + buffer

    # Deterministic filename based on source + times + buffer
    safe_name = Path(audio_file).stem.replace(" ", "_")[:50]
    clip_name = f"{safe_name}_{ss:.1f}_{to:.1f}.mp3"
    clip_path = CLIPS_DIR / clip_name

    if clip_path.exists():
        return clip_path

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(ss),
        "-to", str(to),
        "-i", audio_file,
        "-c", "copy",
        str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return clip_path
    return None


@app.route("/")
def index():
    store = Path(app.config.get("STORE", DEFAULT_STORE))
    cached = list_cached(store)
    return render_template("index.html", transcript_count=len(cached))


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    quote = data.get("quote", "").strip()
    threshold = float(data.get("threshold", 0.70))
    # Client sends 0-100 (e.g. 70), find_quote expects 0-1 (e.g. 0.70)
    if threshold > 1:
        threshold /= 100.0

    if not quote:
        return jsonify({"error": "No quote provided"}), 400

    store = Path(app.config.get("STORE", DEFAULT_STORE))
    matches = _search_all(quote, threshold, store)

    results = []
    for m in matches:
        ctx = m.get("context", {})
        results.append({
            "score": round(m["score"] * 100),
            "audio_name": m["audio_name"],
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "start_fmt": fmt_time(m["start_time"]),
            "end_fmt": fmt_time(m["end_time"]),
            "matched_text": m["matched_text"],
            "context_before": ctx.get("before", ""),
            "context_after": ctx.get("after", ""),
            "context_start": fmt_time(ctx["before_start"]) if "before_start" in ctx else "",
            "context_end": fmt_time(ctx["after_end"]) if "after_end" in ctx else "",
        })

    return jsonify({"quote": quote, "count": len(results), "results": results})


@app.route("/api/clip", methods=["POST"])
def api_clip():
    data = request.get_json()
    audio_name = data.get("audio_name", "")
    start = float(data.get("start_time", 0))
    end = float(data.get("end_time", 0))
    buffer = float(data.get("buffer", 0.3))

    audio_path = _resolve_audio(audio_name)
    if audio_path is None:
        return jsonify({"error": "Audio file not found"}), 404

    clip_path = _make_clip(str(audio_path), start, end, buffer)
    if clip_path is None:
        return jsonify({"error": "Failed to extract clip"}), 500

    return send_file(str(clip_path), mimetype="audio/mpeg")


@app.route("/api/clip_text", methods=["POST"])
def api_clip_text():
    """Search for text and return the best matching clip directly."""
    data = request.get_json()
    quote = data.get("quote", "").strip()
    threshold = float(data.get("threshold", 0.50))
    buffer = float(data.get("buffer", 0.3))
    output_name = data.get("output_name", "").strip()

    if not quote:
        return jsonify({"error": "No text provided"}), 400

    store = Path(app.config.get("STORE", DEFAULT_STORE))
    matches = _search_all(quote, threshold, store)

    if not matches:
        return jsonify({"error": f'No match found for "{quote}"'}), 404

    best = matches[0]
    audio_path = _resolve_audio(best["audio_name"])
    if audio_path is None:
        return jsonify({"error": "Audio file not found on disk"}), 404

    clip_path = _make_clip(str(audio_path), best["start_time"], best["end_time"], buffer)
    if clip_path is None:
        return jsonify({"error": "Failed to extract clip"}), 500

    download_name = None
    if output_name:
        if not output_name.endswith(".mp3"):
            output_name += ".mp3"
        download_name = output_name

    _track_clip(clip_path.name, quote, best["audio_name"], best["start_time"], best["end_time"])
    return send_file(
        str(clip_path),
        mimetype="audio/mpeg",
        as_attachment=bool(download_name),
        download_name=download_name,
    )


@app.route("/api/popular")
def api_popular():
    """Return clips sorted by popularity."""
    stats = _load_clip_stats()
    clips = []
    for clip_name, info in stats.items():
        clip_path = CLIPS_DIR / clip_name
        if clip_path.exists():
            clips.append({
                "clip_name": clip_name,
                "count": info["count"],
                "quote": info.get("quote", ""),
                "audio_name": info.get("audio_name", ""),
                "start": info.get("start", 0),
                "end": info.get("end", 0),
            })
    clips.sort(key=lambda c: c["count"], reverse=True)
    return jsonify(clips)


@app.route("/api/popular/play/<clip_name>")
def api_popular_play(clip_name):
    """Play a clip from the popular list (no count bump)."""
    clip_name = Path(clip_name).name
    clip_path = CLIPS_DIR / clip_name
    if not clip_path.exists():
        return jsonify({"error": "Clip not found"}), 404
    return send_file(str(clip_path), mimetype="audio/mpeg")


@app.route("/api/popular/download/<clip_name>")
def api_popular_download(clip_name):
    """Download a clip from the popular list and bump its count."""
    clip_name = Path(clip_name).name
    clip_path = CLIPS_DIR / clip_name
    if not clip_path.exists():
        return jsonify({"error": "Clip not found"}), 404
    stats = _load_clip_stats()
    if clip_name in stats:
        stats[clip_name]["count"] += 1
        _save_clip_stats(stats)
    return send_file(str(clip_path), mimetype="audio/mpeg", as_attachment=True,
                     download_name=clip_name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audio Quote Finder — Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.config["STORE"] = args.store
    print(f"\nAudio Quote Finder — Web UI")
    print(f"Transcript store: {args.store}")
    print(f"Open http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
