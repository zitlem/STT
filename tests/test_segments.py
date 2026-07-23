"""Word→sentence attribution and audio-type labeling (stt/segments.py)."""

import json

from stt.segments import (
    attribute_words_to_sentences,
    classify_audio_type,
    panns_label_from_prob,
    words_json_or_none,
    words_to_session_ms,
)


class TestPannsLabelFromProb:
    def test_music_above_threshold(self):
        assert panns_label_from_prob(0.8, -20, {}) == "Music"

    def test_speaking_when_audible(self):
        assert panns_label_from_prob(0.2, -20, {}) == "Speaking"

    def test_quiet_when_below_db_threshold(self):
        assert panns_label_from_prob(0.2, -50, {}) == "Quiet"

    def test_missing_db_treated_as_quiet(self):
        assert panns_label_from_prob(0.2, None, {}) == "Quiet"

    def test_custom_thresholds(self):
        cfg = {"music_prob_threshold": 0.9, "quiet_db_threshold": -10}
        assert panns_label_from_prob(0.85, -5, cfg) == "Speaking"
        assert panns_label_from_prob(0.95, -5, cfg) == "Music"
        assert panns_label_from_prob(0.5, -15, cfg) == "Quiet"


class TestClassifyAudioType:
    def test_audible_is_speaking(self):
        assert classify_audio_type(-20, {}) == "Speaking"

    def test_below_threshold_is_quiet(self):
        assert classify_audio_type(-55, {}) == "Quiet"

    def test_missing_db_is_quiet(self):
        assert classify_audio_type(None, {}) == "Quiet"

    def test_never_claims_music(self):
        # Energy fallback must not label anything Music, however loud
        assert classify_audio_type(-1, {}) == "Speaking"

    def test_zero_db_treated_as_missing(self):
        # `audio_db or -60` makes an exact 0 dB reading fall back like None
        assert classify_audio_type(0, {}) == "Quiet"


def word(w, start, end, prob=0.9):
    return {"word": w, "start": start, "end": end, "probability": prob}


class TestWordsToSessionMs:
    def test_no_segments(self):
        assert words_to_session_ms([]) == []
        assert words_to_session_ms(None) == []

    def test_segments_without_words(self):
        # Standard openai-whisper backend emits no words[]
        assert words_to_session_ms([{"text": "hi", "start": 5.0}]) == []

    def test_offset_maps_chunk_time_to_session_time(self):
        # Segment starts at 100s session time; its words are chunk-relative from 2s
        segs = [{"start": 100.0, "words": [word(" hello", 2.0, 2.5), word(" there", 2.6, 3.0)]}]
        stream = words_to_session_ms(segs)
        assert [w["s_ms"] for w in stream] == [100000, 100600]
        assert stream[0]["e_ms"] == 100500
        assert stream[0]["w"] == " hello"
        assert stream[0]["c"] == 0.9

    def test_single_offset_shared_across_batch(self):
        segs = [
            {"start": 100.0, "words": [word("a", 2.0, 2.5)]},
            {"start": 103.0, "words": [word("b", 5.0, 5.5)]},
        ]
        stream = words_to_session_ms(segs)
        # offset = 100 - 2 = 98 applies to both segments' words
        assert [w["s_ms"] for w in stream] == [100000, 103000]

    def test_missing_word_times_default_sanely(self):
        segs = [{"start": 0.0, "words": [{"word": "x", "start": 0.0, "end": None}]}]
        stream = words_to_session_ms(segs)
        assert stream[0]["s_ms"] == 0
        assert stream[0]["e_ms"] == stream[0]["s_ms"]


class TestAttributeWordsToSentences:
    def test_empty_stream(self):
        assert attribute_words_to_sentences([], 3) == [[], [], []]

    def test_single_sentence_gets_all(self):
        stream = [{"w": "a", "s_ms": 0, "e_ms": 100}, {"w": "b", "s_ms": 110, "e_ms": 200}]
        groups = attribute_words_to_sentences(stream, 1)
        assert groups == [stream]

    def test_splits_at_largest_gap(self):
        stream = [
            {"w": "one", "s_ms": 0, "e_ms": 300},
            {"w": "two", "s_ms": 320, "e_ms": 600},      # small gap
            {"w": "three", "s_ms": 2000, "e_ms": 2300},  # big pause -> sentence boundary
            {"w": "four", "s_ms": 2320, "e_ms": 2600},
        ]
        groups = attribute_words_to_sentences(stream, 2)
        assert [w["w"] for w in groups[0]] == ["one", "two"]
        assert [w["w"] for w in groups[1]] == ["three", "four"]

    def test_every_word_lands_exactly_once(self):
        stream = [{"w": str(i), "s_ms": i * 500, "e_ms": i * 500 + 400} for i in range(10)]
        groups = attribute_words_to_sentences(stream, 3)
        flat = [w["w"] for g in groups for w in g]
        assert sorted(flat, key=int) == [str(i) for i in range(10)]
        assert len(groups) == 3

    def test_more_sentences_than_words(self):
        stream = [{"w": "only", "s_ms": 0, "e_ms": 100}]
        groups = attribute_words_to_sentences(stream, 3)
        assert len(groups) == 3
        assert sum(len(g) for g in groups) == 1

    def test_zero_sentences_clamped_to_one(self):
        stream = [{"w": "a", "s_ms": 0, "e_ms": 100}]
        assert attribute_words_to_sentences(stream, 0) == [stream]


class TestWordsJsonOrNone:
    def test_empty_returns_none(self):
        assert words_json_or_none([]) is None
        assert words_json_or_none(None) is None

    def test_round_trip(self):
        objs = [{"w": " hello", "s_ms": 0, "e_ms": 500, "c": 0.98}]
        assert json.loads(words_json_or_none(objs)) == objs

    def test_unicode_not_escaped(self):
        assert '"señor"' in words_json_or_none([{"w": "señor", "s_ms": 0, "e_ms": 1, "c": 1}])

    def test_unserializable_returns_none(self):
        assert words_json_or_none([{"w": object()}]) is None
