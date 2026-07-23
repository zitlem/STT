"""Transcript formatting/export helpers (stt/formatting.py)."""

import json
import sqlite3

import pytest

from stt.formatting import (
    apply_word_highlighting_server,
    convert_db_to_html,
    convert_db_to_srt,
    convert_db_to_translation_srt,
    format_file_size,
    format_timestamp_srt,
    format_timestamp_vtt,
    format_transcription,
)

SEGMENTS = [
    {"text": "Hello world.", "start": 0.0, "end": 2.5},
    {"text": "Second line.", "start": 2.5, "end": 5.0},
]


class TestTimestamps:
    def test_srt_format(self):
        assert format_timestamp_srt(0) == "00:00:00,000"
        assert format_timestamp_srt(3661.5) == "01:01:01,500"

    def test_vtt_format(self):
        assert format_timestamp_vtt(0) == "00:00:00.000"
        assert format_timestamp_vtt(3661.5) == "01:01:01.500"

    def test_millis_truncated_not_rounded(self):
        assert format_timestamp_srt(1.9999).startswith("00:00:01,")


class TestFormatTranscription:
    def test_txt(self):
        assert format_transcription(SEGMENTS, "txt") == "Hello world.\nSecond line."

    def test_srt(self):
        out = format_transcription(SEGMENTS, "srt")
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 2
        assert blocks[0] == "1\n00:00:00,000 --> 00:00:02,500\nHello world."
        assert blocks[1].startswith("2\n")

    def test_vtt(self):
        out = format_transcription(SEGMENTS, "vtt")
        assert out.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:02.500\nHello world." in out

    def test_json(self):
        data = json.loads(format_transcription(SEGMENTS, "json"))
        assert data["total_segments"] == 2
        assert data["duration"] == 5.0
        assert data["segments"][0]["text"] == "Hello world."

    def test_json_empty(self):
        data = json.loads(format_transcription([], "json"))
        assert data == {"segments": [], "total_segments": 0, "duration": 0}

    def test_unknown_format_falls_back_to_text(self):
        assert format_transcription(SEGMENTS, "docx") == "Hello world.\nSecond line."


class TestFormatFileSize:
    def test_units(self):
        assert format_file_size(512) == "512.00 B"
        assert format_file_size(1536) == "1.50 KB"
        assert format_file_size(5 * 1024**3) == "5.00 GB"

    def test_petabytes_cap(self):
        assert format_file_size(3 * 1024**5).endswith(" PB")


class TestApplyWordHighlighting:
    CFG = {"enabled": True, "words": [{"word": "grace", "color": "#ff0000"}]}

    def test_no_config_escapes_only(self):
        assert apply_word_highlighting_server("a <b> & c", None) == "a &lt;b&gt; &amp; c"

    def test_disabled_escapes_only(self):
        cfg = {"enabled": False, "words": [{"word": "x"}]}
        assert apply_word_highlighting_server("x", cfg) == "x"

    def test_highlights_word_case_insensitive(self):
        out = apply_word_highlighting_server("Amazing Grace here", self.CFG)
        assert '<span style="color: #ff0000;">Grace</span>' in out

    def test_word_boundaries(self):
        out = apply_word_highlighting_server("graceful disgrace", self.CFG)
        assert "<span" not in out

    def test_case_sensitive_mode(self):
        cfg = {"enabled": True, "words": [{"word": "Lord", "color": "#0f0", "case_sensitive": True}]}
        assert "<span" not in apply_word_highlighting_server("the lord", cfg)
        assert "<span" in apply_word_highlighting_server("the Lord", cfg)

    def test_regex_mode(self):
        cfg = {"enabled": True, "words": [{"word": r"Psalms? \d+", "color": "#00f", "is_regex": True}]}
        out = apply_word_highlighting_server("see Psalm 23 today", cfg)
        assert '<span style="color: #00f;">Psalm 23</span>' in out

    def test_invalid_regex_skipped(self):
        cfg = {"enabled": True, "words": [{"word": "(", "color": "#00f", "is_regex": True}]}
        assert apply_word_highlighting_server("plain text", cfg) == "plain text"

    def test_disabled_color_group_skipped(self):
        cfg = dict(self.CFG, disabled_colors=["#ff0000"])
        assert "<span" not in apply_word_highlighting_server("grace", cfg)

    def test_input_html_escaped_before_highlighting(self):
        out = apply_word_highlighting_server("<script>grace</script>", self.CFG)
        assert "<script>" not in out
        assert '<span style="color: #ff0000;">grace</span>' in out


ROWS = [
    # (timestamp, text, translated_text, denied, is_final, marked)
    ("not-a-timestamp", "Bad ts row.", "Mal.", 0, 1, 0),           # skipped: unparseable
    ("2026-07-23 10:00:00", "First sentence.", "Primera frase.", 0, 1, 0),
    ("2026-07-23 10:00:02", "Second sentence.", "Segunda frase.", 0, 1, 1),
    ("2026-07-23 10:00:02", "Same-second row.", None, 0, 1, 0),   # end must be >= start+1
    ("2026-07-23 10:00:05", "Denied row.", None, 1, 1, 0),        # excluded
    ("2026-07-23 10:00:06", "Partial row.", None, 0, 0, 0),       # excluded (not final)
    ("2026-07-23 10:00:07", "   ", None, 0, 1, 0),                # excluded (blank)
    ("2026-07-23 10:00:08", "Last sentence.", "Última frase.", 0, 1, 0),
]


@pytest.fixture
def session_db(tmp_path):
    db_path = str(tmp_path / "Transcriptions.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, text TEXT, translated_text TEXT,
                denied INTEGER DEFAULT 0, is_final INTEGER DEFAULT 1,
                marked INTEGER DEFAULT 0)"""
        )
        conn.executemany(
            "INSERT INTO transcriptions (timestamp, text, translated_text, denied, is_final, marked) VALUES (?,?,?,?,?,?)",
            ROWS,
        )
    return db_path


class TestConvertDbToSrt:
    def test_missing_db_returns_none(self, tmp_path):
        assert convert_db_to_srt(str(tmp_path / "nope.db")) is None
        assert convert_db_to_srt(None) is None

    def test_creates_srt_with_filtered_rows(self, session_db, tmp_path):
        srt_path = convert_db_to_srt(session_db, html_enabled=False)
        assert srt_path == session_db.replace(".db", ".srt")
        content = (tmp_path / "Transcriptions.srt").read_text()
        assert "First sentence." in content
        assert "Last sentence." in content
        assert "Denied row." not in content
        assert "Partial row." not in content
        assert "Bad ts row." not in content

    def test_timing_rules(self, session_db, tmp_path):
        convert_db_to_srt(session_db, html_enabled=False)
        content = (tmp_path / "Transcriptions.srt").read_text()
        blocks = content.strip().split("\n\n")
        # First entry ends where the next begins (t=2s)
        assert "00:00:00,000 --> 00:00:02,000" in blocks[0]
        # An entry followed by a same-timestamp row gets a minimum 1s duration
        assert "00:00:02,000 --> 00:00:03,000" in blocks[1]
        # The same-timestamp row itself spans to the next distinct timestamp
        assert "00:00:02,000 --> 00:00:08,000" in blocks[2]
        # Last entry gets start+3s (t=8s -> 11s)
        assert "00:00:08,000 --> 00:00:11,000" in blocks[3]

    def test_html_generated_when_enabled(self, session_db, tmp_path):
        convert_db_to_srt(session_db, html_enabled=True)
        assert (tmp_path / "Transcriptions.html").exists()

    def test_html_skipped_when_disabled(self, session_db, tmp_path):
        convert_db_to_srt(session_db, html_enabled=False)
        assert not (tmp_path / "Transcriptions.html").exists()

    def test_empty_db_returns_none(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, timestamp TEXT, text TEXT, denied INTEGER, is_final INTEGER)")
        assert convert_db_to_srt(db_path) is None


class TestConvertDbToTranslationSrt:
    def test_creates_translated_srt(self, session_db, tmp_path):
        out = convert_db_to_translation_srt(session_db)
        assert out == session_db.replace(".db", ".translated.srt")
        content = (tmp_path / "Transcriptions.translated.srt").read_text()
        assert "Primera frase." in content
        assert "Última frase." in content
        # Rows without a translation are excluded entirely
        assert "Same-second row." not in content

    def test_no_translations_returns_none(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, timestamp TEXT, text TEXT, translated_text TEXT, denied INTEGER, is_final INTEGER)")
            conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES ('2026-07-23 10:00:00', 'untranslated')")
        assert convert_db_to_translation_srt(db_path) is None


class TestConvertDbToHtml:
    def test_missing_db_returns_none(self, tmp_path):
        assert convert_db_to_html(str(tmp_path / "nope.db")) is None

    def test_builds_html_with_marked_badge(self, session_db, tmp_path):
        html_path = convert_db_to_html(session_db)
        assert html_path == session_db.replace(".db", ".html")
        content = (tmp_path / "Transcriptions.html").read_text()
        assert content.startswith("<!DOCTYPE html>")
        assert "First sentence." in content
        assert "Denied row." not in content
        # Exactly one row was marked during the session
        assert content.count('class="segment marked"') == 1
        assert "mark-badge" in content

    def test_old_schema_without_marked_column(self, tmp_path):
        db_path = str(tmp_path / "old.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, timestamp TEXT, text TEXT, denied INTEGER, is_final INTEGER)")
            conn.execute("INSERT INTO transcriptions (timestamp, text) VALUES ('2026-07-23 10:00:00', 'legacy row')")
        html_path = convert_db_to_html(db_path)
        assert html_path is not None
        content = (tmp_path / "old.html").read_text()
        assert "legacy row" in content
        assert 'class="segment marked"' not in content

    def test_highlight_config_applied(self, session_db, tmp_path):
        cfg_path = tmp_path / "word_highlighting.json"
        cfg_path.write_text(json.dumps({"enabled": True, "words": [{"word": "First", "color": "#ff0000"}]}))
        convert_db_to_html(session_db, highlight_config_path=str(cfg_path))
        content = (tmp_path / "Transcriptions.html").read_text()
        assert '<span style="color: #ff0000;">First</span>' in content
