#!/usr/bin/env python3
"""Split an m4b audiobook into per-chapter MP3 files."""
import json
import os
import subprocess
import sys

src = sys.argv[1]
outdir = sys.argv[2]

os.makedirs(outdir, exist_ok=True)

result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", src],
    capture_output=True, text=True
)
chapters = json.loads(result.stdout)["chapters"]
print(f"Found {len(chapters)} chapters in {src}")

for i, ch in enumerate(chapters, 1):
    start = ch["start_time"]
    end = ch["end_time"]
    title = ch.get("tags", {}).get("title", f"Chapter {i}")
    safe_title = title.replace("/", "-").replace(":", "-")
    outfile = os.path.join(outdir, f"{i:02d} - {safe_title}.mp3")
    if os.path.exists(outfile):
        print(f"[skip] {i:02d} - {safe_title}")
        continue
    dur = float(end) - float(start)
    print(f"[{i}/{len(chapters)}] {safe_title} ({dur:.0f}s)")
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ss", start, "-to", end,
        "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-ar", "44100",
        "-loglevel", "error",
        outfile
    ], check=True)

print(f"Done: {len(chapters)} chapters extracted to {outdir}")
