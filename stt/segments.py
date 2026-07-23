"""Segment/word attribution and audio-type labeling logic.

Extracted from speech_to_text.py so it can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only; config
sections are passed in explicitly (thin wrappers in speech_to_text.py
supply the live config).
"""

import json

def panns_label_from_prob(smoothed_prob, audio_db, cfg):
    """Map a (smoothed) music probability to a Speaking/Music/Quiet label."""
    quiet_db_threshold = cfg.get("quiet_db_threshold", -40)
    if smoothed_prob > cfg.get("music_prob_threshold", 0.5):
        return "Music"
    return "Quiet" if (audio_db or -60) <= quiet_db_threshold else "Speaking"


def classify_audio_type(audio_db, cfg):
    """Energy-based fallback label (no PANNs): audible => Speaking, else Quiet.
    We never claim Music without the PANNs detector.

    cfg is the speech_type_detection config section.
    """
    quiet_db_threshold = cfg.get("quiet_db_threshold", -40)
    return "Speaking" if (audio_db or -60) > quiet_db_threshold else "Quiet"


def words_to_session_ms(completed_segments):
    """Flatten faster-whisper word lists from a batch of completed segments into an
    ordered stream of {w, s_ms, e_ms, c} dicts on a session-relative millisecond
    timeline (so words from every row share one timeline for offline replay).

    Word timings are chunk-relative seconds while each segment's `start` is
    session-relative; the segments in one batch share a chunk base, so a single
    offset (the first word-bearing segment's start minus its first word's start)
    maps the whole stream onto the session timeline. `w` is the verbatim
    faster-whisper token (never normalized). Returns [] when no segment carries
    word data (e.g. the standard openai-whisper backend, which emits no words[]).
    """
    segs = [s for s in (completed_segments or []) if s.get('words')]
    if not segs:
        return []
    fw = segs[0]['words']
    try:
        off = (segs[0].get('start') or 0) - (fw[0].get('start') or 0)
    except Exception:
        off = 0
    stream = []
    for seg in segs:
        for w in seg.get('words', []):
            ws, we = w.get('start'), w.get('end')
            ws = ws if ws is not None else 0
            we = we if we is not None else ws
            stream.append({
                "w": w.get('word', ''),
                "s_ms": round((ws + off) * 1000),
                "e_ms": round((we + off) * 1000),
                "c": w.get('probability'),
            })
    return stream


def attribute_words_to_sentences(stream, num_sentences):
    """Assign each word in an ordered session-ms `stream` to one of `num_sentences`
    re-split rows by MAX TEMPORAL OVERLAP, so no boundary word is ever dropped
    (re-splits fall on pauses, exactly where book tokens sit).

    Provisional per-sentence spans are found by cutting the stream at its
    `num_sentences-1` largest inter-word gaps (the pauses the sentence splitter
    used) — robust without relying on token text. Each word is then assigned to the
    span it overlaps most; ties or gap-words go to the earlier sentence. Returns a
    list of `num_sentences` word-lists; every word lands in exactly one list.
    """
    n = max(1, num_sentences)
    groups = [[] for _ in range(n)]
    if not stream:
        return groups
    if n == 1:
        groups[0] = list(stream)
        return groups
    # Cut at the largest inter-word gaps -> provisional contiguous groups.
    gaps = [(stream[i]["s_ms"] - stream[i - 1]["e_ms"], i) for i in range(1, len(stream))]
    cut_count = min(n - 1, len(gaps))
    cut_idx = sorted(i for _, i in sorted(gaps, key=lambda g: g[0], reverse=True)[:cut_count])
    prov, start = [], 0
    for ci in cut_idx:
        prov.append(stream[start:ci])
        start = ci
    prov.append(stream[start:])
    while len(prov) < n:
        prov.append([])
    prov = prov[:n]
    spans = [((g[0]["s_ms"], g[-1]["e_ms"]) if g else None) for g in prov]

    def _overlap(w, span):
        if span is None:
            return None
        return min(w["e_ms"], span[1]) - max(w["s_ms"], span[0])

    for w in stream:
        best_j, best_ov = 0, None
        for j, span in enumerate(spans):
            ov = _overlap(w, span)
            if ov is None:
                continue
            if best_ov is None or ov > best_ov:
                best_ov, best_j = ov, j
        groups[best_j].append(w)
    return groups


def words_json_or_none(word_objs):
    """Serialize a list of {w, s_ms, e_ms, c} word dicts to a JSON array string for
    the `words_json` column, or None when empty (NULL = no per-word data — see
    `words_source` to tell 'backend has no words' from an alignment miss)."""
    if not word_objs:
        return None
    try:
        return json.dumps(word_objs, ensure_ascii=False)
    except Exception:
        return None
