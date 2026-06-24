#!/usr/bin/env python3
"""
segmenter.py  —  Step 2 of 3
==============================
Reads transcript.txt (output of transcriber.py), segments it into
coherent topics using a single Groq LLM call per chunk, and attaches
real timestamps to each segment.

For long transcripts, chunks are processed independently and their
results are simply concatenated in order — no overlap, no merge logic.

Output:
    segments.json  — array of topic segments, each with:
        {
          "index":      0,
          "title":      "...",
          "summary":    "...",
          "start_time": "00:01:23",
          "end_time":   "00:04:11",
          "text":       "..."
        }

Requirements:
    pip install groq python-dotenv
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
#  CONFIG
# ============================================================
TRANSCRIPT_FILE     = "transcript.txt"
OUTPUT_FILE         = "segments.json"
GROQ_MODEL          = "llama-3.3-70b-versatile"
MAX_LINES_PER_CHUNK = 250   # lines sent to the LLM in one call
# ============================================================


SEGMENTATION_PROMPT = """\
You are an expert transcript analyst. You will receive a portion of a transcript
where every line is prefixed with its ORIGINAL line number, like:

120: [00:04:05] Hello everyone.
121: [00:04:08] Today we discuss the budget.

Your task:
1. Segment this portion into coherent topic sections in chronological order.
2. Give each segment a concise title (8 words or less).
3. Write a 1-3 sentence summary for each segment.
4. Return the START and END line numbers using the ORIGINAL numbers shown above.

Return ONLY a valid JSON array — no markdown fences, no commentary:
[
  { "title": "...", "summary": "...", "start_line": 120, "end_line": 135 },
  ...
]

Rules:
- Use the exact line numbers shown in the input — do NOT renumber from 0.
- Every line in this portion must be covered — no gaps.
- First segment starts at the first line shown; last segment ends at the last line shown.
- Produce as many segments as needed — do not merge unrelated topics.
- IMPORTANT: The transcript is in Arabic. You MUST write ALL titles and summaries in Arabic. Do NOT use English under any circumstances.
"""


TIMESTAMP_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]")


def parse_lines(transcript: str) -> list[str]:
    return [l for l in transcript.splitlines() if l.strip()]


def extract_timestamp(line: str) -> str | None:
    m = TIMESTAMP_RE.match(line.strip())
    return m.group(1) if m else None


def clean_line(line: str) -> str:
    return TIMESTAMP_RE.sub("", line).strip()


def parse_json_list(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def numbered_chunk(lines: list[str], start_idx: int, end_idx: int) -> str:
    """Return lines[start_idx..end_idx] each prefixed with its original index."""
    return "\n".join(
        f"{i}: {lines[i]}" for i in range(start_idx, min(end_idx + 1, len(lines)))
    )


def segment_chunk(lines: list[str], start_idx: int, end_idx: int, client: Groq) -> list[dict]:
    """
    Send lines[start_idx..end_idx] to the LLM for segmentation.
    Returns a list of dicts with start_line/end_line in original coordinates.
    """
    chunk_text = numbered_chunk(lines, start_idx, end_idx)

    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SEGMENTATION_PROMPT},
            {"role": "user",   "content": chunk_text},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    data = parse_json_list(r.choices[0].message.content)
    data = sorted(data, key=lambda x: x["start_line"])

    # Clamp line numbers to the actual chunk boundaries (model may hallucinate)
    for seg in data:
        seg["start_line"] = max(start_idx, seg["start_line"])
        seg["end_line"]   = min(end_idx,   seg["end_line"])

    return data


def attach_timestamps(raw_segs: list[dict], lines: list[str]) -> list[dict]:
    """
    Convert start_line/end_line to real timestamps from the transcript,
    collect the segment text, and build the final output format.
    """
    total_lines = len(lines)
    segments    = []

    for i, item in enumerate(raw_segs):
        start         = max(0, item.get("start_line", 0))
        end           = min(total_lines - 1, item.get("end_line", start))
        segment_lines = lines[start : end + 1]
        text          = " ".join(clean_line(l) for l in segment_lines if clean_line(l))

        start_time = next(
            (extract_timestamp(l) for l in segment_lines if extract_timestamp(l)),
            "00:00:00",
        )
        end_time = next(
            (extract_timestamp(l) for l in reversed(segment_lines) if extract_timestamp(l)),
            start_time,
        )

        segments.append({
            "index":      i,
            "title":      item.get("title", f"Topic {i + 1}"),
            "summary":    item.get("summary", ""),
            "start_time": start_time,
            "end_time":   end_time,
            "text":       text,
        })

    return segments


def segment_transcript(transcript: str, client: Groq) -> list[dict]:
    """
    Full pipeline:
    1. Split transcript into chunks of MAX_LINES_PER_CHUNK.
    2. Segment each chunk with one LLM call.
    3. Concatenate results and attach timestamps.
    """
    lines       = parse_lines(transcript)
    total_lines = len(lines)

    # Build non-overlapping chunk ranges
    ranges = []
    start  = 0
    while start < total_lines:
        end = min(start + MAX_LINES_PER_CHUNK - 1, total_lines - 1)
        ranges.append((start, end))
        start = end + 1

    print(f"  Transcript: {total_lines} lines → {len(ranges)} chunk(s) "
          f"(max {MAX_LINES_PER_CHUNK} lines each).")

    all_raw_segs = []
    for i, (cs, ce) in enumerate(ranges):
        print(f"  Segmenting chunk {i + 1}/{len(ranges)} "
              f"(lines {cs}–{ce})...", end="", flush=True)
        segs = segment_chunk(lines, cs, ce, client)
        print(f" {len(segs)} segment(s).")
        all_raw_segs.extend(segs)

    # Re-index and attach timestamps
    segments = attach_timestamps(all_raw_segs, lines)
    for i, seg in enumerate(segments):
        seg["index"] = i

    return segments


def print_results(segments: list[dict]) -> None:
    print("\n" + "=" * 70)
    for seg in segments:
        print(f"\n  [{seg['index'] + 1}] {seg['title']}")
        print(f"       {seg['start_time']} → {seg['end_time']}")
        print(f"       {seg['summary']}")
    print("\n" + "=" * 70)


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        raise SystemExit(1)

    script_dir      = Path(__file__).parent
    transcript_path = script_dir / TRANSCRIPT_FILE
    if not transcript_path.exists():
        print(f"Error: {transcript_path} not found. Run transcriber.py first.")
        raise SystemExit(1)

    print(f"\n=== Step 2: Segmentation ===\n")
    print(f"  Reading: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8")

    client   = Groq(api_key=groq_api_key)
    segments = segment_transcript(transcript, client)

    print_results(segments)

    output_path = script_dir / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"\n  Segments saved → {output_path}")
    print(f"\n=== Done. Run describer.py next. ===\n")


if __name__ == "__main__":
    main()
