"""Testable slices of stt/audio_capture.py: device parsing/resolution and
ffmpeg command construction. The capture loop itself (subprocess/mic) is out
of unit-test scope."""

import os
import sys

from stt.audio_capture import (
    FFmpegAudioCapture,
    create_compatible_audio_source,
    parse_asound_cards,
    resolve_audio_device_by_name,
)

CARDS = """ 0 [NVidia         ]: HDA-Intel - HDA NVidia
                      HDA NVidia at 0xfcffc000 irq 22
 1 [USB            ]: USB-Audio - Blue Yeti USB Microphone
                      Blue Microphones Blue Yeti at usb-0000:00:14.0-2
"""


class TestParseAsoundCards:
    def test_parses_cards_with_plughw_names(self):
        devices = parse_asound_cards(CARDS)
        assert [d["name"] for d in devices] == ["plughw:0,0", "plughw:1,0"]
        # The type/description split happens at the FIRST hyphen, so a hyphenated
        # driver name ("HDA-Intel") leaks its tail into the display name.
        assert devices[0]["display_name"] == "Intel - HDA NVidia"
        assert devices[1]["card_id"] == "USB"

    def test_first_device_default_when_none_deprioritized(self):
        devices = parse_asound_cards(CARDS)
        assert [d["is_default"] for d in devices] == [True, False]

    def test_deprioritized_first_card_yields_second_default(self):
        # HDMI/GPU audio should not win the default slot over a real mic
        devices = parse_asound_cards(CARDS, deprioritize_markers=["nvidia"])
        assert [d["is_default"] for d in devices] == [False, True]

    def test_all_deprioritized_falls_back_to_first(self):
        devices = parse_asound_cards(CARDS, deprioritize_markers=["nvidia", "usb"])
        assert [d["is_default"] for d in devices] == [True, False]

    def test_internal_flag_stripped_from_output(self):
        for d in parse_asound_cards(CARDS, deprioritize_markers=["nvidia"]):
            assert "is_deprioritized" not in d

    def test_empty_or_garbage_content(self):
        assert parse_asound_cards("") == []
        assert parse_asound_cards("no cards here\njust noise") == []


DEVICES = [
    {"name": "plughw:0,0", "card_id": "NVidia", "display_name": "HDA NVidia"},
    {"name": "plughw:1,0", "card_id": "USB", "display_name": "Blue Yeti USB Microphone"},
]


class TestResolveDeviceByName:
    def test_matches_display_name_substring_case_insensitive(self):
        dev = resolve_audio_device_by_name("blue yeti", DEVICES)
        assert dev["name"] == "plughw:1,0"

    def test_matches_card_id(self):
        assert resolve_audio_device_by_name("NVidia", DEVICES)["name"] == "plughw:0,0"

    def test_card_id_contained_in_saved_name(self):
        # e.g. saved "USB Audio Device" matches card_id "USB"
        assert resolve_audio_device_by_name("USB Audio Device", DEVICES)["name"] == "plughw:1,0"

    def test_no_match_or_empty(self):
        assert resolve_audio_device_by_name("nonexistent mic", DEVICES) is None
        assert resolve_audio_device_by_name("", DEVICES) is None
        assert resolve_audio_device_by_name("   ", DEVICES) is None
        assert resolve_audio_device_by_name(None, DEVICES) is None


class TestInit:
    def test_chunk_size_and_sample_width(self):
        cap = FFmpegAudioCapture(sample_rate=16000, chunk_duration=0.5, ts_enabled=False)
        assert cap.chunk_size == 8000
        assert cap.SAMPLE_WIDTH == 2
        assert cap.SAMPLE_RATE == 16000

    def test_ts_disabled_has_no_backup_dir(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert cap.backup_dir is None

    def test_explicit_backup_dir_honored(self, tmp_path):
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), ts_enabled=True)
        assert cap.backup_dir == str(tmp_path)

    def test_filename_defaults(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert cap.filename_format == "%Y-%m-%d_%H%M%S"
        assert cap.filename_prefix == ""

    def test_flush_buffer_sets_event(self):
        cap = FFmpegAudioCapture(ts_enabled=False)
        assert not cap._flush_event.is_set()
        cap.flush_buffer()
        assert cap._flush_event.is_set()


class TestGetFfmpegCommand:
    def test_backup_file_named_from_format_and_prefix(self, tmp_path):
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), filename_prefix="sunday", ts_enabled=True)
        cap.device_name = None
        cmd = cap._get_ffmpeg_command()
        assert cap.backup_file.startswith(str(tmp_path) + os.sep)
        assert cap.backup_file.endswith("_sunday.ts")
        assert cmd[-1] == cap.backup_file  # backup file is the mpegts output

    def test_backup_dir_created(self, tmp_path):
        backup = tmp_path / "2026" / "07"
        cap = FFmpegAudioCapture(backup_dir=str(backup), ts_enabled=True)
        cap._get_ffmpeg_command()
        assert backup.is_dir()

    def test_ts_split_counter_increments(self, tmp_path):
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), ts_enabled=True)
        cap._get_ffmpeg_command()
        cap._get_ffmpeg_command()
        assert cap._ts_file_count == 2

    def test_file_playback_mode(self, tmp_path):
        wav = tmp_path / "input.wav"
        wav.write_bytes(b"\0")
        cap = FFmpegAudioCapture(device_name=str(wav), ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert cmd[:3] == ["ffmpeg", "-y", "-re"]  # -re: real-time pacing for file input
        assert str(wav) in cmd
        assert "s16le" in cmd and "pipe:1" in cmd

    def test_mic_mode_no_ts_is_pcm_only(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        cap = FFmpegAudioCapture(device_name="plughw:1,0", ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert ["-f", "alsa", "-i", "plughw:1,0"] == cmd[2:6]
        assert "mpegts" not in cmd
        assert "-filter_complex" not in cmd

    def test_mic_mode_with_ts_splits_to_backup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        cap = FFmpegAudioCapture(backup_dir=str(tmp_path), ts_enabled=True)
        cmd = cap._get_ffmpeg_command()
        assert "alsa" in cmd
        assert "-filter_complex" in cmd
        assert "mpegts" in cmd and cmd[-1] == cap.backup_file

    def test_darwin_device_gets_colon_prefix(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        cap = FFmpegAudioCapture(device_name="1", ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert ["-f", "avfoundation", "-i", ":1"] == cmd[2:6]

    def test_windows_device_gets_audio_prefix(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cap = FFmpegAudioCapture(device_name="Blue Yeti", ts_enabled=False)
        cmd = cap._get_ffmpeg_command()
        assert ["-f", "dshow", "-i", "audio=Blue Yeti"] == cmd[2:6]


class TestCreateCompatibleAudioSource:
    def test_source_has_queue_and_interface_attrs(self):
        src = create_compatible_audio_source(sample_rate=8000, ts_enabled=False)
        assert src.SAMPLE_RATE == 8000
        assert src.SAMPLE_WIDTH == 2
        assert src.data_queue is not None
        assert src.data_queue.empty()
