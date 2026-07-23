"""Pure text-processing helpers for the transcription/translation pipeline.

Extracted from speech_to_text.py so they can be imported (and unit-tested)
without the monolith's import-time side effects. Functions here must stay
stdlib-only and free of module-level state other than diagnostics counters;
anything that needs runtime config takes it as an explicit parameter, with
thin wrappers in speech_to_text.py supplying the live config.
"""

import re
from difflib import SequenceMatcher


def filter_hallucinated_text(text, language="auto"):
    """
    Filter out hallucinated foreign characters when transcribing specific languages.
    Whisper sometimes hallucinates random Chinese/Japanese/Korean characters
    in the middle of English transcriptions.

    Args:
        text: The transcribed text
        language: The target language code (e.g., "en", "auto")

    Returns:
        Filtered text with foreign characters removed if language is specific
    """
    if not text:
        return text

    # Remove CJK (Chinese, Japanese, Korean) characters
    # Unicode ranges:
    # - CJK Unified Ideographs: \u4e00-\u9fff
    # - Hiragana: \u3040-\u309f
    # - Katakana: \u30a0-\u30ff
    # - Hangul: \uac00-\ud7af
    cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+'

    filtered = re.sub(cjk_pattern, '', text)

    # Clean up any double spaces left behind
    filtered = re.sub(r'\s+', ' ', filtered).strip()

    if filtered != text:
        print(f"[FILTER] Removed hallucinated characters: '{text}' -> '{filtered}'")

    return filtered


# Default Whisper hallucinations - phantom subtitle credits from training data
# These are common phrases Whisper hallucinates from YouTube videos in its training set
# Matching is substring-based (phrase anywhere in the sentence, case/punctuation insensitive),
# so one entry catches all surface variants (e.g. "DimaTorzok" catches all 3 Russian credit lines).
# User can override/extend this list via config.json hallucination_filter.phrases
DEFAULT_WHISPER_HALLUCINATIONS = [
    "DimaTorzok",           # catches all Субтитры * DimaTorzok variants
    "Продолжение следует",
    "for watching",         # catches "thank you for watching", "thanks for watching", etc.
    "Please subscribe",
    "Like and subscribe",
    "Don't forget to subscribe",
]


def get_hallucination_phrases(hallucination_config):
    """Get hallucination phrases from the hallucination_filter config section, falling back to defaults."""
    if not hallucination_config.get("enabled", True):
        return []  # Filter disabled
    return hallucination_config.get("phrases", DEFAULT_WHISPER_HALLUCINATIONS)


def normalize_for_hallucination_check(text):
    """Normalize text for hallucination comparison: lowercase, remove apostrophes and punctuation."""
    if not text:
        return ""
    # Lowercase and strip
    normalized = text.strip().lower()
    # Remove apostrophes (both straight and curly)
    normalized = normalized.replace("'", "").replace("’", "").replace("‘", "")
    # Remove punctuation but keep spaces and alphanumeric
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def apply_profanity_filter(text, cfg):
    """Replace broadcast-forbidden words with **** (or configured replacement).

    cfg is the profanity_filter config section.
    """
    if not text:
        return text
    if not cfg.get("enabled", False):
        return text
    words = cfg.get("words", [])
    if not words:
        return text
    replacement = cfg.get("replacement", "****")
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def is_whisper_hallucination(text, phrases):
    """Check if text is a known Whisper hallucination (exact phrase match, case/punctuation insensitive)."""
    if not text:
        return False

    if not phrases:
        return False  # Filter disabled or empty

    text_normalized = normalize_for_hallucination_check(text)
    for hallucination in phrases:
        if normalize_for_hallucination_check(hallucination) in text_normalized:
            return True
    return False


def split_into_sentences(text):
    """
    Split text into complete sentences and remainder.

    Returns:
        tuple: (list of complete sentences, remaining incomplete text)
    """
    if not text:
        return [], ""

    sentences = []
    remainder = text.strip()

    # Pattern: sentence ending with . ! ? followed by space or end
    # Also handles multiple punctuation like "..." or "!?"
    pattern = r'^(.*?[.!?]+)(?:\s+|$)'

    while remainder:
        match = re.match(pattern, remainder)
        if match:
            sentence = match.group(1).strip()
            if sentence:  # Only add non-empty sentences
                sentences.append(sentence)
            remainder = remainder[match.end():].strip()
        else:
            break  # No more complete sentences

    return sentences, remainder


def count_sentence_units(text):
    """Count sentence units in text (a trailing incomplete fragment counts as one)."""
    sentences, remainder = split_into_sentences(text)
    return len(sentences) + (1 if remainder else 0)


_ctx_align_stats = {"exact": 0, "proportional": 0, "failed": 0}


def extract_context_translation(combined_translated, num_context_sentences, source_char_ratio=None):
    """
    Extract the target portion from a translation of (context + target) text.

    Splits the combined translation into sentences and drops the first
    num_context_sentences. When the translator merged sentences (fewer output
    units than context units) and source_char_ratio is given (context chars /
    total source chars), falls back to splitting at the sentence boundary whose
    cumulative character share is closest to that ratio — sentence boundaries
    stay intact, and the context benefit isn't discarded. Returns None only
    when no sensible split exists, so the caller can retranslate the target
    in isolation.
    """
    if num_context_sentences <= 0:
        return combined_translated
    if not combined_translated:
        return None
    out_sentences, out_remainder = split_into_sentences(combined_translated)
    if out_remainder:
        out_sentences = out_sentences + [out_remainder]
    if len(out_sentences) > num_context_sentences:
        _ctx_align_stats["exact"] += 1
        _log_ctx_align_stats()
        return " ".join(out_sentences[num_context_sentences:]).strip() or None
    if source_char_ratio is not None and len(out_sentences) >= 2:
        total_len = max(1, len(combined_translated))
        best_k, best_diff = 1, None
        cum = 0
        for k in range(1, len(out_sentences)):
            cum += len(out_sentences[k - 1]) + 1
            diff = abs(cum / total_len - source_char_ratio)
            if best_diff is None or diff < best_diff:
                best_k, best_diff = k, diff
        _ctx_align_stats["proportional"] += 1
        _log_ctx_align_stats()
        return " ".join(out_sentences[best_k:]).strip() or None
    _ctx_align_stats["failed"] += 1
    _log_ctx_align_stats()
    return None


def _log_ctx_align_stats():
    """Log alignment mix every 50 extractions so the mismatch rate is measurable."""
    total = sum(_ctx_align_stats.values())
    if total and total % 50 == 0:
        print(f"[CTX-ALIGN] exact={_ctx_align_stats['exact']} proportional={_ctx_align_stats['proportional']} failed={_ctx_align_stats['failed']}", flush=True)


def distribute_whisper_translation(translated_text, row_texts):
    """
    Split a whole-chunk Whisper translation across the rows saved from that chunk.

    Pass 2 of whisper_translate translates the entire audio chunk in one call,
    but that chunk may have produced several DB rows. Sentence-count alignment
    is used when it matches; otherwise the translation is split into word spans
    proportional to each source row's length. Callers first narrow the text to
    the batch's time span via scope_whisper_translation; the split here is
    still approximate when sentence counts don't match.

    Returns a list of len(row_texts) strings.
    """
    n = len(row_texts)
    if n == 0:
        return []
    if n == 1:
        return [translated_text]

    units, rem = split_into_sentences(translated_text)
    if rem.strip():
        units.append(rem.strip())
    if len(units) == n:
        return units

    words = translated_text.split()
    if len(words) < n:
        # Too little text to split meaningfully; keep it on the first row
        return [translated_text] + [""] * (n - 1)

    src_counts = [max(1, len(t.split())) for t in row_texts]
    total = sum(src_counts)
    parts = []
    start = 0
    for i, count in enumerate(src_counts):
        if i == n - 1:
            end = len(words)
        else:
            end = start + max(1, round(len(words) * count / total))
            # Leave at least one word for every remaining row
            end = min(end, len(words) - (n - 1 - i))
            end = max(end, start + 1)
        parts.append(" ".join(words[start:end]))
        start = end
    return parts


def scope_whisper_translation(pass2_timed, span_end, margin=0.5):
    """
    Keep only pass-2 translation segments that fall within the finalized span.

    Pass 2 of whisper_translate covers the whole unfinalized buffer, including
    the still-in-progress tail; that tail's translation belongs to future rows,
    not this batch. Dropping segments whose midpoint lies past span_end keeps
    the distributed translation aligned with the rows actually being saved.

    pass2_timed: list of (session_start, session_end, text).
    Returns the joined in-span text, or None if nothing/nothing usable remains
    (caller falls back to the full pass-2 text).
    """
    if not pass2_timed or span_end is None:
        return None
    parts = [t for (s, e, t) in pass2_timed if (s + e) / 2 <= span_end + margin]
    return " ".join(parts).strip() or None


def is_fuzzy_duplicate(new_sentence, saved_sentences, threshold=0.85):
    """
    Check if new_sentence is a fuzzy duplicate of any saved sentence.

    Uses difflib.SequenceMatcher for similarity comparison.

    Args:
        new_sentence: The sentence to check
        saved_sentences: List of already-saved sentences
        threshold: Minimum similarity ratio (0.0-1.0) to consider duplicate

    Returns:
        True if new_sentence is a fuzzy duplicate of any saved sentence
    """
    if not new_sentence or not saved_sentences:
        return False

    new_lower = new_sentence.lower().strip()

    for saved in saved_sentences:
        saved_lower = saved.lower().strip()
        ratio = SequenceMatcher(None, new_lower, saved_lower).ratio()
        if ratio >= threshold:
            return True

    return False


def remove_overlapping_prefix(new_text, previous_text, min_overlap_words=3):
    """
    Remove overlapping prefix from new_text if it matches the end of previous_text.

    This handles the case where Whisper's rolling buffer transcription produces
    text that starts with the end of the previous finalized text.

    Example:
        previous: "I hope you are doing well from your food comas"
        new: "from your food comas that you experienced"
        result: "that you experienced"

    Args:
        new_text: The new text to check
        previous_text: The previously saved text
        min_overlap_words: Minimum words to consider as overlap (default 3)

    Returns:
        new_text with overlapping prefix removed, or original if no overlap
    """
    if not new_text or not previous_text:
        return new_text

    new_words = new_text.split()
    prev_words = previous_text.split()

    if len(new_words) < min_overlap_words or len(prev_words) < min_overlap_words:
        return new_text

    # Check if new_text starts with the end of previous_text
    # Start with longest possible overlap and work down
    max_overlap = min(len(new_words), len(prev_words))
    for overlap_len in range(max_overlap, min_overlap_words - 1, -1):
        # Get last N words of previous and first N words of new
        prev_suffix = ' '.join(prev_words[-overlap_len:]).lower()
        new_prefix = ' '.join(new_words[:overlap_len]).lower()

        # Use fuzzy match to handle minor transcription differences
        if SequenceMatcher(None, prev_suffix, new_prefix).ratio() > 0.85:
            # Found overlap - return text after the overlap
            remaining = ' '.join(new_words[overlap_len:])
            if remaining.strip():
                print(f"[OVERLAP] Removed {overlap_len} words: '{new_prefix[:40]}...'", flush=True)
                return remaining
            else:
                # Entire new text was overlap - return empty
                print("[OVERLAP] Entire text was overlap, skipping", flush=True)
                return ""

    return new_text
