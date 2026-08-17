"""Strip sponsor reads from raw transcript segments before chunking.

Podcast ad reads are transcribed like any other speech, so without this they
are chunked, embedded and returned as retrieval hits: asking about Shaolin
discipline surfaces a LinkedIn hiring pitch, and every such hit costs a slot
in the top-K that real content could have used.

Filtering happens on raw segments rather than on chunks, because it is the
one point that feeds both the flat and parent-document indexes -- filtering
later would let them disagree about what the corpus contains.

Detection is deliberately conservative. Deleting a sponsor read is a small
win; deleting real content is a silent, unrecoverable loss, so a span has to
be triggered by a marker that does not plausibly occur in ordinary speech.
Run this module directly for a dry run over the fetched transcripts:

    python -m ingestion.ad_filter
"""
import json
import os
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_transcripts"

# One of these is enough to open an ad span. Every pattern here was checked
# against the actual corpus; anything that also matched ordinary speech was
# demoted or dropped. Notably absent:
#   "terms and conditions" -- fired 8 times on a genuine discussion about
#       scam checkout terms, and never on an actual ad.
#   bare "sponsor"         -- matches disclosures ("today's episode does
#       include sponsors") and outros ("check out the sponsors mentioned"),
#       neither of which is a read.
STRONG_MARKERS = [
    r"\bour sponsors?\b",
    r"\bsponsored by\b",
    r"\bbrought to you by\b",
    r"\bthis episode is sponsored\b",
    r"\buse (?:the )?(?:promo |discount )?code\b",
    r"\b(?:promo|discount|coupon) code\b",
    r"\b\d{1,3}\s*(?:%|percent) off\b",
    r"\bfree trial\b",
]

# Vocabulary that keeps an already-open span open. Never used to trigger one:
# individually these are ordinary words, and several ("support", "quality",
# "try") are common in health and self-help talk, which is most of this corpus.
#
# The body of an ad read carries almost no hard markers -- it is plain product
# prose ("AG1 is designed to support things like gut health") -- so the span
# has to be held open by the density of this softer vocabulary instead.
PROMO_MARKERS = [
    r"\bcheckout\b",
    r"\bsign up\b",
    r"\blimited time\b",
    r"\blink in the (?:description|show notes|bio)\b",
    r"\bvisit\b",
    r"\border\b",
    r"\bsave\b",
    r"\bdeal\b",
    r"\bshop\b",
    r"\bdiscount\b",
    r"\bsubscription\b",
    r"\bspecial offer\b",
    r"\bfirst order\b",
    r"\bgo to\b",
    r"\bcustomers?\b",
    r"\bproducts?\b",
    r"\bwww\.|\.com\b|\.co\b",
    # Spoken URLs: reads spell them out ("J-O-O-V-V dot com slash huberman"),
    # so the literal ".com" above never matches the sign-off line.
    r"\bdot com\b|\bdot co\b",
    r"\bslash \w+\b",
    r"\bfree\b",
    r"\btry (?:it|them|out)\b",
    r"\bhighest quality\b",
    r"\bi'?ve been (?:using|taking|sleeping|drinking)\b",
    r"\bi (?:personally )?(?:take|use|drink)\b",
]

# Cues that an ad read is being announced, used only to extend a span
# backwards to its real start -- the marker usually lands mid-read.
LEAD_IN_MARKERS = [
    r"\b(?:quick|short) break\b",
    r"\bbefore we (?:continue|get back|carry on)\b",
    r"\bi'?d like to (?:take|thank|acknowledge)\b",
    r"\ba word from\b",
]

STRONG_RE = re.compile("|".join(STRONG_MARKERS), re.I)
PROMO_RE = re.compile("|".join(PROMO_MARKERS), re.I)
LEAD_IN_RE = re.compile("|".join(LEAD_IN_MARKERS), re.I)

# Ad blocks chain -- an AG1 read runs straight into an 8Sleep read -- so the
# cap has to allow a few minutes. It exists only to stop a false trigger from
# swallowing an episode, not to bound a typical read.
MAX_SPAN_SECONDS = 300.0
MAX_LEAD_IN_SECONDS = 45.0

# The span stays open while the next LOOKAHEAD segments carry at least
# MIN_PROMO_HITS promotional signals. Judging a window rather than each
# segment is what lets a read survive its own ordinary sentences.
LOOKAHEAD_SEGMENTS = 6
MIN_PROMO_HITS = 2
# How far the sign-off sweep may run past the window's verdict.
TRAILING_SWEEP_SEGMENTS = 8


def strip_ads_enabled() -> bool:
    """On by default; STRIP_ADS=0 disables it."""
    return os.environ.get("STRIP_ADS", "1") != "0"


def _seg_end(seg: dict) -> float:
    return seg.get("start", 0.0) + seg.get("duration", 0.0)


def _promo_score(text: str) -> int:
    """Distinct promotional markers in one segment (distinct, so a repeated
    word does not on its own look like a dense pitch)."""
    return len(set(m.group(0).lower() for m in PROMO_RE.finditer(text)))


# The advertiser's name, taken from the announcement itself. Matches "our
# sponsor, AG1", "brought to us by 8Sleep", "sponsored by Oscars VCs".
BRAND_RE = re.compile(
    r"(?:our sponsors?|sponsored by|brought to (?:you|us) by)[,:\s]+"
    r"([A-Za-z0-9][\w'&.-]*(?:\s+[A-Z][\w'&.-]*)?)",
    re.I,
)
# Words that follow the announcement but are not the brand.
_BRAND_STOPWORDS = {"the", "a", "an", "my", "our", "this", "today", "one", "of"}


def _extract_brands(segments: list[dict], start: int, end: int) -> set[str]:
    brands = set()
    for seg in segments[start:end]:
        for m in BRAND_RE.finditer(seg.get("text", "")):
            for token in m.group(1).split():
                token = token.strip(".,!?'\"").lower()
                if len(token) > 2 and token not in _BRAND_STOPWORDS:
                    brands.add(token)
    return brands


def _window_score(segments: list[dict], start: int, brands: set[str]) -> int:
    """Promotional signal in the next LOOKAHEAD_SEGMENTS segments.

    Brand repetition carries the most weight: the body of a read is ordinary
    prose apart from naming the product in nearly every sentence, which is the
    one thing that reliably distinguishes it from the surrounding interview.
    """
    score = 0
    for seg in segments[start:start + LOOKAHEAD_SEGMENTS]:
        text = seg.get("text", "")
        score += _promo_score(text)
        if STRONG_RE.search(text):
            score += 2
        lowered = text.lower()
        if any(re.search(rf"\b{re.escape(b)}\b", lowered) for b in brands):
            score += 2
    return score


def find_ad_spans(segments: list[dict]) -> list[tuple[int, int]]:
    """Return [start_index, end_index) segment ranges that look like ad reads."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(segments):
        if not STRONG_RE.search(segments[i].get("text", "")):
            i += 1
            continue

        # Walk backwards to the announcement, so "I'd like to take a quick
        # break and acknowledge our sponsor" is removed whole rather than
        # leaving its first half stranded in the corpus.
        start = i
        while start > 0:
            prev = segments[start - 1]
            if segments[i]["start"] - prev.get("start", 0.0) > MAX_LEAD_IN_SECONDS:
                break
            text = prev.get("text", "")
            if LEAD_IN_RE.search(text) or _promo_score(text) >= 1 or STRONG_RE.search(text):
                start -= 1
            else:
                break

        # Walk forwards while the surrounding window still reads as a pitch.
        # Scoring a window rather than the current segment is what carries the
        # span across an ad's ordinary sentences. Brands are re-read as the
        # span grows so a chained read (AG1 straight into 8Sleep) is caught by
        # the second advertiser's name too.
        end = i + 1
        brands = _extract_brands(segments, start, end + LOOKAHEAD_SEGMENTS)
        while end < len(segments):
            if _seg_end(segments[end]) - segments[start].get("start", 0.0) > MAX_SPAN_SECONDS:
                break
            if _window_score(segments, end, brands) < MIN_PROMO_HITS:
                break
            end += 1
            brands |= _extract_brands(segments, end, end + LOOKAHEAD_SEGMENTS)

        # The window straddles the boundary as the read winds down, so it drops
        # below threshold while sign-off lines are still to come ("Again,
        # that's functionhealth.com slash huberman"). Sweep those up one at a
        # time, on per-segment evidence only, bounded so a miss cannot run on.
        swept = 0
        while end < len(segments) and swept < TRAILING_SWEEP_SEGMENTS:
            text = segments[end].get("text", "")
            lowered = text.lower()
            if not (STRONG_RE.search(text)
                    or _promo_score(text) >= 1
                    or any(re.search(rf"\b{re.escape(b)}\b", lowered) for b in brands)):
                break
            end += 1
            swept += 1

        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
        i = end

    return spans


def strip_ads(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split segments into (kept, removed). Timestamps are untouched, so the
    deep links built from surviving segments still point at the right moment."""
    spans = find_ad_spans(segments)
    if not spans:
        return segments, []

    dropped = {i for start, end in spans for i in range(start, end)}
    kept = [s for i, s in enumerate(segments) if i not in dropped]
    removed = [s for i, s in enumerate(segments) if i in dropped]
    return kept, removed


def _fmt(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def main():
    """Dry run: report what would be stripped, without touching anything."""
    paths = sorted(RAW_DIR.glob("*.json"))
    if not paths:
        print(f"No transcripts in {RAW_DIR}")
        return

    for path in paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        segments = doc["segments"]
        spans = find_ad_spans(segments)
        total = sum(_seg_end(segments[e - 1]) - segments[s]["start"] for s, e in spans)
        episode_seconds = _seg_end(segments[-1]) if segments else 0.0

        print(f"\n{doc['video_id']}  {len(spans)} span(s), "
              f"{total / 60:.1f} min of {episode_seconds / 60:.0f} min "
              f"({total / episode_seconds * 100 if episode_seconds else 0:.1f}%)")
        for s, e in spans:
            text = " ".join(seg.get("text", "") for seg in segments[s:e])
            print(f"  {_fmt(segments[s]['start'])}-{_fmt(_seg_end(segments[e - 1]))} "
                  f"({e - s} segs): {text[:150]}...")


if __name__ == "__main__":
    main()
