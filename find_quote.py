#!/usr/bin/env python3
"""Audio Quote Finder — transcribe once, search many times, extract clips.

Transcripts are cached as JSON in a local store directory so audio files
only need to be transcribed once. Subsequent searches are instant.

Subcommands:
    transcribe  Transcribe audio file(s) and cache the results
    search      Search cached transcripts for a quote
    clip        Extract an audio clip given a file and time range
    list        Show all cached transcripts

Usage:
    python find_quote.py transcribe *.mp3
    python find_quote.py search "what the hell are you doing"
    python find_quote.py clip audio.mp3 12:15.2 12:17.1 -o quote.mp3

Requirements:
    pip install openai-whisper rapidfuzz
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

try:
    import whisper
except ImportError:
    whisper = None
from rapidfuzz import fuzz

DEFAULT_STORE = Path(__file__).parent / "transcripts"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".m4b", ".flac", ".ogg", ".wma", ".aac", ".aax", ".aa", ".opus"}


# ── Transcript store ──────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    """Fast hash: file size + first/last 64KB + mtime."""
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
        if stat.st_size > 65536:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    return h.hexdigest()[:16]


def _store_path(store_dir: Path, audio_path: Path, base_dir: Path | None = None) -> Path:
    """Return the JSON path for a given audio file's transcript.

    If base_dir is provided, the transcript mirrors the relative path
    from base_dir under store_dir.  Otherwise it's flat in store_dir.
    """
    safe_name = re.sub(r"[^\w\-.]]", "_", audio_path.stem)
    fhash = _file_hash(audio_path)
    fname = f"{safe_name}_{fhash}.json"

    if base_dir is not None:
        try:
            rel = audio_path.resolve().parent.relative_to(base_dir.resolve())
            return store_dir / rel / fname
        except ValueError:
            pass
    return store_dir / fname


def load_transcript(store_dir: Path, audio_path: Path, base_dir: Path | None = None) -> dict | None:
    """Load a cached transcript, or None if not cached."""
    p = _store_path(store_dir, audio_path, base_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_transcript(store_dir: Path, audio_path: Path, data: dict, base_dir: Path | None = None):
    """Save a transcript to the store."""
    p = _store_path(store_dir, audio_path, base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def list_cached(store_dir: Path) -> list[dict]:
    """List all cached transcripts (searches recursively)."""
    results = []
    if not store_dir.exists():
        return results
    for p in sorted(store_dir.rglob("*.json")):
        with open(p) as f:
            data = json.load(f)
        results.append(
            {
                "file": data.get("audio_file", "?"),
                "words": len(data.get("words", [])),
                "model": data.get("model", "?"),
                "cache_file": str(p.relative_to(store_dir)),
            }
        )
    return results


# ── Whisper transcription ─────────────────────────────────────────────


def transcribe_audio(audio_path: str, model_name: str = "base") -> list[dict]:
    """Transcribe audio and return word-level timestamps."""
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed. Install it with: pip install openai-whisper")
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing '{audio_path}' (this may take a while)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )

    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
            )
    return words


def get_or_transcribe(
    audio_path: Path, store_dir: Path, model_name: str = "base",
    force: bool = False, base_dir: Path | None = None,
) -> list[dict]:
    """Return words for an audio file, transcribing only if not cached."""
    if not force:
        cached = load_transcript(store_dir, audio_path, base_dir)
        if cached is not None:
            print(f"  [cached] {audio_path.name} ({len(cached['words'])} words)")
            return cached["words"]

    words = transcribe_audio(str(audio_path), model_name)
    save_transcript(
        store_dir,
        audio_path,
        {
            "audio_file": str(audio_path.resolve()),
            "model": model_name,
            "words": words,
        },
        base_dir,
    )
    print(f"  [saved]  {audio_path.name} ({len(words)} words)")
    return words


# ── Search logic ──────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def find_quote(
    words: list[dict], quote: str, threshold: float = 0.70
) -> list[dict]:
    """Two-pass fuzzy search: exact word scan → fuzzy match candidates only."""
    quote_norm = normalize(quote)
    quote_words = quote_norm.split()
    quote_word_count = len(quote_words)

    if quote_word_count == 0:
        return []

    # Pre-normalize all words once
    norm_words = [normalize(w["word"]) for w in words]
    n = len(norm_words)

    min_win = max(1, int(quote_word_count * 0.7))
    max_win = int(quote_word_count * 1.4) + 1

    # ── Pass 1: find candidate regions via exact word hits ──
    query_word_set = set(quote_words)
    # For each position, record whether it's a hit
    hits = [1 if nw in query_word_set else 0 for nw in norm_words]

    # Use a sliding window the size of max_win to count hits in each region.
    # A region is a candidate if it contains enough query words.
    # Require at least 30% of query words to be present (very loose to avoid
    # missing fuzzy matches from transcription errors).
    min_hits = max(1, int(quote_word_count * 0.3))
    candidate_positions = set()

    # Sliding sum over max_win-sized regions
    scan_win = max_win
    if n >= scan_win:
        window_hits = sum(hits[:scan_win])
        if window_hits >= min_hits:
            for j in range(scan_win):
                candidate_positions.add(j)
        for i in range(1, n - scan_win + 1):
            window_hits += hits[i + scan_win - 1] - hits[i - 1]
            if window_hits >= min_hits:
                # Mark all positions in this region as candidates
                for j in range(i, i + scan_win):
                    candidate_positions.add(j)

    # If no candidates found, fall back to full scan (handles heavy
    # transcription errors where no exact words match)
    full_scan = len(candidate_positions) == 0

    # ── Pass 2: fuzzy match only candidate windows ──
    cutoff = int(threshold * 100)
    matches = []
    for win_size in range(min_win, max_win + 1):
        for i in range(n - win_size + 1):
            # Skip if this window doesn't overlap any candidate region
            if not full_scan and i not in candidate_positions:
                continue

            window_text = " ".join(norm_words[i : i + win_size])
            score = fuzz.ratio(quote_norm, window_text, score_cutoff=cutoff)
            if score > 0:
                matches.append(
                    {
                        "score": score / 100.0,
                        "start_time": words[i]["start"],
                        "end_time": words[i + win_size - 1]["end"],
                        "matched_text": " ".join(
                            w["word"] for w in words[i : i + win_size]
                        ),
                    }
                )

    matches.sort(key=lambda m: m["score"], reverse=True)
    filtered = []
    for m in matches:
        overlaps = False
        for kept in filtered:
            if m["start_time"] < kept["end_time"] and m["end_time"] > kept["start_time"]:
                overlaps = True
                break
        if not overlaps:
            filtered.append(m)

    return filtered


# ── Formatting ────────────────────────────────────────────────────────


def fmt_time(seconds: float) -> str:
    """Format seconds as H:MM:SS.s or M:SS.s."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:04.1f}"
    return f"{m}:{s:04.1f}"


def parse_time(t: str) -> float:
    """Parse M:SS.s or H:MM:SS.s or bare seconds into float seconds."""
    parts = t.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


# ── Subcommands ───────────────────────────────────────────────────────


def cmd_transcribe(args):
    """Transcribe one or more audio files and cache the results."""
    store = Path(args.store)
    paths = []
    base_dir = None
    for p in args.audio:
        p = Path(p)
        if p.is_dir():
            base_dir = p.resolve()
            paths.extend(
                sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
            )
        else:
            paths.append(p)

    if not paths:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    # Load model once for the whole batch
    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)

    print(f"Transcribing {len(paths)} file(s)...\n")
    for i, audio_path in enumerate(paths, 1):
        if not audio_path.exists():
            print(f"  [skip]   {audio_path} (not found)")
            continue
        print(f"[{i}/{len(paths)}]", end=" ")
        # Check cache first
        if not args.force:
            cached = load_transcript(store, audio_path, base_dir)
            if cached is not None:
                print(f"  [cached] {audio_path.name} ({len(cached['words'])} words)")
                continue
        # Transcribe using the already-loaded model
        print(f"  Transcribing {audio_path.name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                verbose=False,
            )
        words = []
        for segment in result["segments"]:
            for w in segment.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })
        save_transcript(
            store, audio_path,
            {"audio_file": str(audio_path.resolve()), "model": args.model, "words": words},
            base_dir,
        )
        print(f"         [saved]  {audio_path.name} ({len(words)} words)")

    print(f"\nDone. Transcripts stored in: {store.resolve()}")


def _get_context(words: list[dict], start_time: float, end_time: float,
                  context_words: int = 30) -> dict:
    """Get surrounding context words for a match.

    Returns dict with 'before', 'match', 'after' text and their time ranges.
    """
    # Find indices of the matched region
    match_start_idx = None
    match_end_idx = None
    for i, w in enumerate(words):
        if match_start_idx is None and w["start"] >= start_time - 0.01:
            match_start_idx = i
        if w["end"] <= end_time + 0.01:
            match_end_idx = i

    if match_start_idx is None:
        match_start_idx = 0
    if match_end_idx is None:
        match_end_idx = len(words) - 1

    before_start = max(0, match_start_idx - context_words)
    after_end = min(len(words), match_end_idx + 1 + context_words)

    before_words = words[before_start:match_start_idx]
    after_words = words[match_end_idx + 1:after_end]

    return {
        "before": " ".join(w["word"] for w in before_words),
        "before_start": before_words[0]["start"] if before_words else start_time,
        "match": " ".join(w["word"] for w in words[match_start_idx:match_end_idx + 1]),
        "after": " ".join(w["word"] for w in after_words),
        "after_end": after_words[-1]["end"] if after_words else end_time,
    }


def cmd_search(args):
    """Search cached transcripts for a quote."""
    store = Path(args.store)
    if not store.exists():
        print("No transcripts cached yet. Run 'transcribe' first.", file=sys.stderr)
        sys.exit(1)

    transcript_files = sorted(store.rglob("*.json"))
    if not transcript_files:
        print("No transcripts cached yet. Run 'transcribe' first.", file=sys.stderr)
        sys.exit(1)

    all_matches = []
    threshold = args.threshold if args.threshold <= 1 else args.threshold / 100.0
    for tf in transcript_files:
        with open(tf) as f:
            data = json.load(f)
        audio_file = data.get("audio_file", tf.stem)
        words = data.get("words", [])
        matches = find_quote(words, args.quote, threshold)
        for m in matches:
            m["audio_file"] = audio_file
            m["context"] = _get_context(words, m["start_time"], m["end_time"],
                                        context_words=args.context)
        all_matches.extend(matches)

    all_matches.sort(key=lambda m: m["score"], reverse=True)

    if not all_matches:
        print(f"No match found for: \"{args.quote}\"")
        print(f"Searched {len(transcript_files)} transcript(s).")
        print("Try lowering --threshold or using a larger model when transcribing.")
        sys.exit(1)

    print(f"Found {len(all_matches)} match(es) for: \"{args.quote}\"")
    print(f"Searched {len(transcript_files)} transcript(s).\n")
    for i, m in enumerate(all_matches, 1):
        display_name = Path(m["audio_file"]).name
        ctx = m["context"]
        print(f"  Match {i}  (score: {m['score']:.0%})")
        print(f"    File:  {display_name}")
        print(f"    Time:  {fmt_time(m['start_time'])} → {fmt_time(m['end_time'])}")
        print()
        print(f"    ...{ctx['before']}")
        print(f"    >>> {ctx['match']} <<<")
        print(f"    {ctx['after']}...")
        print()
        print(f"    Context range: {fmt_time(ctx['before_start'])} → {fmt_time(ctx['after_end'])}")
        print(f"    To clip:  python find_quote.py clip \"{m['audio_file']}\" {fmt_time(m['start_time'])} {fmt_time(m['end_time'])}")
        if args.clip:
            _extract_clip(
                m["audio_file"],
                m["start_time"],
                m["end_time"],
                buffer=args.buffer,
            )
        print()


def cmd_clip(args):
    """Extract an audio clip from a file."""
    start = parse_time(args.start)
    end = parse_time(args.end)
    _extract_clip(args.audio, start, end, buffer=args.buffer, output=args.output)


def _extract_clip(
    audio_file: str,
    start: float,
    end: float,
    buffer: float = 0.3,
    output: str | None = None,
):
    """Extract a clip using ffmpeg."""
    audio_path = Path(audio_file)
    if output is None:
        safe = re.sub(r"[^\w\-.]", "_", audio_path.stem)
        output = f"clip_{safe}_{fmt_time(start).replace(':', '-')}.mp3"

    ss = max(0, start - buffer)
    to = end + buffer

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(ss),
        "-to", str(to),
        "-i", str(audio_path),
        "-c", "copy",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"    Clip:  {output}")
    else:
        print(f"    Clip FAILED: {result.stderr[:200]}", file=sys.stderr)


def cmd_list(args):
    """List all cached transcripts."""
    store = Path(args.store)
    cached = list_cached(store)
    if not cached:
        print("No transcripts cached yet.")
        return

    print(f"{len(cached)} cached transcript(s):\n")
    for c in cached:
        display_name = Path(c["file"]).name
        print(f"  {display_name}")
        print(f"    Words: {c['words']}  Model: {c['model']}")
        print()


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Audio Quote Finder — transcribe once, search many times."
    )
    parser.add_argument(
        "--store",
        default=str(DEFAULT_STORE),
        help=f"Transcript store directory (default: {DEFAULT_STORE})",
    )
    sub = parser.add_subparsers(dest="command")

    # transcribe
    p_tr = sub.add_parser("transcribe", help="Transcribe audio file(s)")
    p_tr.add_argument(
        "audio", nargs="+", help="Audio files or directories to transcribe"
    )
    p_tr.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    p_tr.add_argument(
        "--force", action="store_true", help="Re-transcribe even if cached"
    )

    # search
    p_se = sub.add_parser("search", help="Search transcripts for a quote")
    p_se.add_argument("quote", help="The text quote to search for")
    p_se.add_argument(
        "--threshold",
        type=float,
        default=70,
        help="Minimum fuzzy match score 0-100 (default: 70)",
    )
    p_se.add_argument(
        "--clip", action="store_true", help="Auto-extract clips for each match"
    )
    p_se.add_argument(
        "--buffer",
        type=float,
        default=0.3,
        help="Seconds of buffer around clips (default: 0.3)",
    )
    p_se.add_argument(
        "--context",
        type=int,
        default=30,
        help="Number of words of context on each side (default: 30)",
    )

    # clip
    p_cl = sub.add_parser("clip", help="Extract a clip from an audio file")
    p_cl.add_argument("audio", help="Audio file path")
    p_cl.add_argument("start", help="Start time (e.g. 12:15.2)")
    p_cl.add_argument("end", help="End time (e.g. 12:17.1)")
    p_cl.add_argument("-o", "--output", help="Output filename")
    p_cl.add_argument(
        "--buffer",
        type=float,
        default=0.3,
        help="Seconds of buffer on each side (default: 0.3)",
    )

    # list
    sub.add_parser("list", help="List cached transcripts")

    args = parser.parse_args()

    if args.command == "transcribe":
        cmd_transcribe(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "clip":
        cmd_clip(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
