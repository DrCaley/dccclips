# Audio Quote Finder

Find the exact timestamp when a quote is spoken in an audio file.

Uses [OpenAI Whisper](https://github.com/openai/whisper) for transcription with word-level timestamps, then fuzzy-matches your query against the transcript.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python find_quote.py audio.mp3 "the quote you're looking for"
```

### Options

| Flag | Description |
|------|-------------|
| `--model {tiny,base,small,medium,large}` | Whisper model size (default: `base`). Larger = slower but more accurate. |
| `--threshold 0.7` | Minimum fuzzy match score 0–1 (default: `0.70`). Lower to find looser matches. |
| `--show-transcript` | Print the full word-by-word transcript with timestamps. |

### Examples

```bash
# Basic search
python find_quote.py interview.mp3 "we need to rethink the approach"

# Use a more accurate model
python find_quote.py lecture.wav "gradient descent" --model medium

# Looser matching for paraphrased quotes
python find_quote.py podcast.mp3 "something about neural networks" --threshold 0.5

# See the full transcript
python find_quote.py speech.mp3 "any phrase" --show-transcript
```

### Output

```
Found 1 match(es) for: "we need to rethink the approach"

  Match 1  (score: 94%)
    Time:  12:34.2 → 12:37.8
    Text:  "we need to rethink the approach"
```
