"""Inverted index for fast fuzzy quote search across all transcripts.

Loads all transcripts once at startup, pre-normalizes words, and builds
a word → transcript mapping so searches only scan transcripts that
contain query words.
"""

import json
import time
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from find_quote import normalize, _get_context

# Long queries blow up: 30 words → 22 window sizes × thousands of positions.
# Step through window sizes + time limit keeps searches bounded.
_SEARCH_TIME_LIMIT = 12.0   # seconds
_MAX_RESULTS = 50           # stop scanning once we have enough good results


class SearchIndex:
    """In-memory inverted index over pre-loaded transcripts.

    Only keeps normalized words in memory (strings). Raw word dicts with
    timestamps are loaded on-demand for matching transcripts to save RAM
    on the 1GB OCI VM.
    """

    def __init__(self, store: Path):
        self.transcripts: list[dict] = []   # [{audio_file, norm_words, path}]
        self.index: dict[str, list[int]] = {}  # norm_word → [tidx, ...]
        self.transcript_count = 0
        self._build(store)

    def _build(self, store: Path):
        if not store.exists():
            return
        for tf in sorted(store.rglob("*.json")):
            with open(tf) as f:
                data = json.load(f)
            words = data.get("words", [])
            norm_words = [normalize(w["word"]) for w in words]
            tidx = len(self.transcripts)
            self.transcripts.append({
                "audio_file": data.get("audio_file", "?"),
                "norm_words": norm_words,
                "path": str(tf),
            })
            # Index unique words per transcript (not per position)
            for nw in set(norm_words):
                if nw:
                    self.index.setdefault(nw, []).append(tidx)
        self.transcript_count = len(self.transcripts)

    def _load_words(self, tidx: int) -> list[dict]:
        """Load raw word dicts (with timestamps) for a transcript."""
        with open(self.transcripts[tidx]["path"]) as f:
            return json.load(f).get("words", [])

    def search(self, quote: str, threshold: float = 0.70) -> list[dict]:
        """Search all indexed transcripts for a quote using fuzzy matching.

        Returns matches sorted by score, each with audio_file, audio_name,
        matched_text, start_time, end_time, score, and context.
        """
        quote_norm = normalize(quote)
        quote_words = quote_norm.split()
        count = len(quote_words)
        if count == 0:
            return []

        query_word_set = set(quote_words)
        min_hits = max(1, int(count * 0.3))
        min_win = max(1, int(count * 0.7))
        max_win = int(count * 1.4) + 1
        cutoff = int(threshold * 100)

        # For long queries, step through window sizes to limit work
        win_range = max_win - min_win + 1
        if win_range > 8:
            step = max(1, win_range // 6)
            win_sizes = list(range(min_win, max_win + 1, step))
            # Always include the exact-length window
            if count not in win_sizes:
                win_sizes.append(count)
                win_sizes.sort()
        else:
            win_sizes = list(range(min_win, max_win + 1))

        # Find transcripts containing enough unique query words
        transcript_hits = Counter()
        for qw in query_word_set:
            for tidx in self.index.get(qw, []):
                transcript_hits[tidx] += 1

        # Search best-matching transcripts first (most unique query words)
        ranked = sorted(transcript_hits.items(), key=lambda x: x[1], reverse=True)

        all_matches = []
        t_start = time.monotonic()

        for tidx, unique_hit_count in ranked:
            if unique_hit_count < min_hits:
                break  # sorted desc, so all remaining are below threshold
            if (time.monotonic() - t_start) > _SEARCH_TIME_LIMIT:
                break

            t = self.transcripts[tidx]
            norm_words = t["norm_words"]
            n = len(norm_words)

            # ── Pass 1: candidate regions via exact word hits ──
            hits = [1 if nw in query_word_set else 0 for nw in norm_words]

            candidate_positions = set()
            scan_win = max_win
            if n >= scan_win:
                window_hits = sum(hits[:scan_win])
                if window_hits >= min_hits:
                    for j in range(scan_win):
                        candidate_positions.add(j)
                for i in range(1, n - scan_win + 1):
                    window_hits += hits[i + scan_win - 1] - hits[i - 1]
                    if window_hits >= min_hits:
                        for j in range(i, i + scan_win):
                            candidate_positions.add(j)

            full_scan = len(candidate_positions) == 0

            # ── Pass 2: fuzzy match candidate windows ──
            # First pass: find matching positions using only norm_words
            hit_positions = []  # (score, start_idx, win_size)
            for win_size in win_sizes:
                if win_size > n:
                    continue
                for i in range(n - win_size + 1):
                    if not full_scan and i not in candidate_positions:
                        continue
                    window_text = " ".join(norm_words[i : i + win_size])
                    score = fuzz.ratio(quote_norm, window_text, score_cutoff=cutoff)
                    if score > 0:
                        hit_positions.append((score / 100.0, i, win_size))

            if not hit_positions:
                continue

            # Load raw words with timestamps only if we have matches
            words = self._load_words(tidx)
            matches = []
            for score, i, win_size in hit_positions:
                matches.append({
                    "score": score,
                    "start_time": words[i]["start"],
                    "end_time": words[i + win_size - 1]["end"],
                    "matched_text": " ".join(
                        w["word"] for w in words[i : i + win_size]
                    ),
                })

            # Filter overlapping results within this transcript
            matches.sort(key=lambda m: m["score"], reverse=True)
            for m in matches:
                overlaps = False
                for kept in all_matches:
                    if (kept.get("audio_file") == t["audio_file"]
                            and m["start_time"] < kept["end_time"]
                            and m["end_time"] > kept["start_time"]):
                        overlaps = True
                        break
                if not overlaps:
                    m["audio_file"] = t["audio_file"]
                    m["audio_name"] = Path(t["audio_file"]).name
                    m["context"] = _get_context(
                        words, m["start_time"], m["end_time"]
                    )
                    all_matches.append(m)

        all_matches.sort(key=lambda m: m["score"], reverse=True)
        return all_matches[:_MAX_RESULTS]
