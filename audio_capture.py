"""
Audio capture module using ffmpeg backend only.
PyAudio/ALSA dependencies have been removed - ffmpeg handles everything.
"""

import subprocess
import threading
import time
import sys
import os
import numpy as np
from queue import Queue
from typing import Optional, Callable
from datetime import datetime

# select.select() doesn't work on pipes on Windows - only on sockets
_IS_WINDOWS = sys.platform.startswith('win')
if not _IS_WINDOWS:
    import select


class FFmpegAudioCapture:
    """Audio capture using ffmpeg - reliable cross-platform audio backend"""

    def __init__(self, sample_rate=16000, chunk_duration=1.0, device_name=None, backup_dir=None, filename_format=None, filename_prefix=None, ts_enabled=True):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device_name = device_name
        self.process = None
        self.thread = None
        self.running = False
        self.data_queue = None  # Queue for compatibility with speech_recognition interface
        # Uppercase attributes for compatibility with speech_recognition interface
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = 2  # 16-bit = 2 bytes per sample
        # Buffer flush support for phrase timeout
        self._flush_event = threading.Event()
        # Audio backup file path (for power-fail recovery)
        self.backup_file = None
        # Track how many .ts files have been created this session (for split detection)
        self._ts_file_count = 0
        # Whether .ts backup is enabled
        self.ts_enabled = ts_enabled
        # Filename format from config (default: %Y-%m-%d_%H%M%S)
        self.filename_format = filename_format if filename_format else "%Y-%m-%d_%H%M%S"
        # Filename prefix from config (default: empty)
        self.filename_prefix = filename_prefix if filename_prefix else ""
        # Use provided backup_dir or default to _AUTOMATIC_BACKUP with date path
        if backup_dir and ts_enabled:
            self.backup_dir = backup_dir
        elif ts_enabled:
            # Default: _AUTOMATIC_BACKUP/YYYY/MM (same as full session audio)
            # Use local timezone for path
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_AUTOMATIC_BACKUP")
            date_path = datetime.now().astimezone().strftime("%Y/%m")
            self.backup_dir = os.path.join(base_dir, date_path)
        else:
            self.backup_dir = None

        # Debug: Log initialization settings
        if self.ts_enabled:
            print(f"[DEBUG-TS-INIT] Backup dir: {self.backup_dir}", flush=True)
            print(f"[DEBUG-TS-INIT] Filename format: {self.filename_format}, prefix: '{self.filename_prefix}'", flush=True)
        else:
            print(f"[DEBUG-TS-INIT] .ts backup DISABLED", flush=True)

    def flush_buffer(self):
        """Signal capture thread to flush any remaining buffered audio data.

        Called when phrase timeout triggers to ensure partial audio chunks
        get sent to the queue immediately (padded with silence if needed).
        """
        self._flush_event.set()

    def _get_ffmpeg_command(self):
        """Build ffmpeg command for the current platform with optional MPEG-TS backup"""
        import sys

        # Create backup directory and generate timestamped filename (only if ts_enabled)
        if self.ts_enabled:
            os.makedirs(self.backup_dir, exist_ok=True)
            # Use local timezone for timestamp (astimezone() gets system local time)
            timestamp = datetime.now().astimezone().strftime(self.filename_format)
            # Build filename: {timestamp}_{prefix}.ts or {timestamp}.ts
            if self.filename_prefix:
                self.backup_file = os.path.join(self.backup_dir, f"{timestamp}_{self.filename_prefix}.ts")
            else:
                self.backup_file = os.path.join(self.backup_dir, f"{timestamp}.ts")

            self._ts_file_count += 1
            print(f"[DEBUG-TS-FILE] Created backup path: {self.backup_file}", flush=True)
            if self._ts_file_count > 1:
                print(f"[DEBUG-TS-SPLIT] *** WARNING: THIS IS BACKUP FILE #{self._ts_file_count} - RECORDING HAS BEEN SPLIT ***", flush=True)

        if sys.platform.startswith('linux'):
            # Linux: Use ALSA or PulseAudio
            if self.device_name:
                device = self.device_name
            else:
                device = 'default'  # ALSA default device

            if self.ts_enabled:
                # Use filter_complex to split audio: raw PCM to stdout, MP2 to MPEG-TS file
                # MPEG-TS requires a proper codec (mp2) for power-fail recovery
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'alsa',
                    '-i', device,
                    '-ar', str(self.sample_rate),
                    '-ac', '1',  # mono
                    '-filter_complex', '[0:a]asplit=2[a1][a2]',
                    '-map', '[a1]', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1',
                    '-map', '[a2]', '-c:a', 'mp2', '-b:a', '128k', '-f', 'mpegts', self.backup_file
                ]
            else:
                # Simple command without .ts backup - just PCM to stdout
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'alsa',
                    '-i', device,
                    '-ar', str(self.sample_rate),
                    '-ac', '1',  # mono
                    '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'
                ]

        elif sys.platform == 'darwin':
            # macOS: Use avfoundation
            if self.device_name:
                device = f':{self.device_name}'
            else:
                device = ':0'  # default device

            if self.ts_enabled:
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'avfoundation',
                    '-i', device,
                    '-ar', str(self.sample_rate),
                    '-ac', '1',
                    '-filter_complex', '[0:a]asplit=2[a1][a2]',
                    '-map', '[a1]', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1',
                    '-map', '[a2]', '-c:a', 'mp2', '-b:a', '128k', '-f', 'mpegts', self.backup_file
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'avfoundation',
                    '-i', device,
                    '-ar', str(self.sample_rate),
                    '-ac', '1',
                    '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'
                ]

        elif sys.platform.startswith('win'):
            # Windows: Use dshow
            if self.device_name:
                device = f'audio={self.device_name}'
            else:
                device = 'audio=Microphone'

            if self.ts_enabled:
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'dshow',
                    '-i', device,
                    '-ar', str(self.sample_rate),
                    '-ac', '1',
                    '-filter_complex', '[0:a]asplit=2[a1][a2]',
                    '-map', '[a1]', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1',
                    '-map', '[a2]', '-c:a', 'mp2', '-b:a', '128k', '-f', 'mpegts', self.backup_file
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'dshow',
                    '-i', device,
                    '-ar', str(self.sample_rate),
                    '-ac', '1',
                    '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'
                ]
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        print(f"[DEBUG-TS-CMD] FFmpeg command: {' '.join(cmd)}", flush=True)
        return cmd

    def _capture_loop(self, callback):
        """Main capture loop running in a separate thread"""
        # Debug tracking variables
        bytes_received_total = 0
        last_debug_time = time.time()

        try:
            cmd = self._get_ffmpeg_command()
            print(f"[FFMPEG] Starting audio capture: {' '.join(cmd)}")

            # Start ffmpeg process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.chunk_size * 2  # 16-bit samples
            )

            print(f"[DEBUG-TS-START] FFmpeg started, PID: {self.process.pid}", flush=True)
            print(f"[DEBUG-TS-START] Backup file target: {self.backup_file}", flush=True)

            # Start thread to capture stderr for debugging
            def log_stderr():
                try:
                    for line in self.process.stderr:
                        decoded = line.decode().strip()
                        if decoded:
                            print(f"[DEBUG-TS-STDERR] {decoded}", flush=True)
                except Exception as e:
                    print(f"[DEBUG-TS-STDERR] Error reading stderr: {e}", flush=True)
            stderr_thread = threading.Thread(target=log_stderr, daemon=True)
            stderr_thread.start()

            bytes_per_chunk = self.chunk_size * 2  # 16-bit = 2 bytes per sample
            timeout_count = 0
            buffer = b''  # Buffer for accumulating partial reads

            # Windows: select() doesn't work on pipes, use a threaded reader instead
            pipe_queue = None
            pipe_eof = None
            reader_thread = None

            def _pipe_reader(stdout, q, eof_event):
                """Read from pipe in a blocking thread, push data to queue"""
                try:
                    while True:
                        data = stdout.read(bytes_per_chunk)
                        if not data:
                            break
                        q.put(data)
                except (OSError, ValueError):
                    pass
                finally:
                    eof_event.set()

            if _IS_WINDOWS:
                pipe_queue = Queue()
                pipe_eof = threading.Event()
                reader_thread = threading.Thread(
                    target=_pipe_reader,
                    args=(self.process.stdout, pipe_queue, pipe_eof),
                    daemon=True,
                )
                reader_thread.start()
            else:
                # Unix: use select() + non-blocking I/O for efficient pipe reading
                os.set_blocking(self.process.stdout.fileno(), False)

            def _restart_ffmpeg():
                """Kill and restart ffmpeg, returns new buffer (empty bytes)"""
                nonlocal cmd, pipe_queue, pipe_eof, reader_thread
                old_backup_file = self.backup_file
                print(f"[DEBUG-TS-RESTART] Old backup file: {old_backup_file}", flush=True)
                if old_backup_file and os.path.exists(old_backup_file):
                    print(f"[DEBUG-TS-RESTART] Old backup file size: {os.path.getsize(old_backup_file)} bytes", flush=True)
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                cmd = self._get_ffmpeg_command()
                print(f"[DEBUG-TS-SPLIT] *** BACKUP FILE SPLIT DUE TO TIMEOUT *** Old: {old_backup_file} -> New: {self.backup_file}", flush=True)
                print(f"[FFMPEG] Restarting: {' '.join(cmd)}", flush=True)
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=self.chunk_size * 2
                )
                print(f"[DEBUG-TS-RESTART] New FFmpeg PID: {self.process.pid}", flush=True)
                if _IS_WINDOWS:
                    pipe_queue = Queue()
                    pipe_eof = threading.Event()
                    reader_thread = threading.Thread(
                        target=_pipe_reader,
                        args=(self.process.stdout, pipe_queue, pipe_eof),
                        daemon=True,
                    )
                    reader_thread.start()
                else:
                    os.set_blocking(self.process.stdout.fileno(), False)
                return b''

            while self.running:
                # --- Wait for data with timeout ---
                if _IS_WINDOWS:
                    # Windows: pull from threaded reader queue
                    try:
                        data = pipe_queue.get(timeout=2.0)
                        has_data = True
                    except Exception:
                        data = None
                        has_data = False
                    if not has_data and pipe_eof.is_set() and pipe_queue.empty():
                        print(f"[DEBUG-TS-DIED] FFmpeg pipe closed, exit code: {self.process.poll()}", flush=True)
                        break
                else:
                    # Unix: use select to wait for data with timeout
                    try:
                        ready, _, _ = select.select([self.process.stdout], [], [], 2.0)
                    except (ValueError, OSError) as e:
                        print(f"[FFMPEG] Select error: {e}", flush=True)
                        break
                    has_data = bool(ready)
                    data = None

                if not has_data:
                    # Timeout - check if process is still alive
                    timeout_count += 1
                    print(f"[DEBUG-TS-TIMEOUT] No data for {timeout_count * 2}s, process poll: {self.process.poll()}", flush=True)
                    if self.process.poll() is not None:
                        print(f"[DEBUG-TS-DIED] FFmpeg died with exit code: {self.process.returncode}", flush=True)
                        if self.backup_file and os.path.exists(self.backup_file):
                            print(f"[DEBUG-TS-DIED] Backup file size at death: {os.path.getsize(self.backup_file)} bytes", flush=True)
                        else:
                            print(f"[DEBUG-TS-DIED] WARNING: Backup file does not exist: {self.backup_file}", flush=True)
                        break
                    if timeout_count >= 5:  # 10 seconds of no data
                        print(f"[DEBUG-TS-RESTART] Restarting FFmpeg after {timeout_count * 2}s timeout", flush=True)
                        buffer = _restart_ffmpeg()
                        timeout_count = 0
                    continue

                # Data available - read it
                timeout_count = 0  # Reset timeout counter
                try:
                    if _IS_WINDOWS:
                        # data already read from queue above
                        if not data:
                            print("[FFMPEG] No more audio data (EOF)", flush=True)
                            break
                    else:
                        data = os.read(self.process.stdout.fileno(), bytes_per_chunk)
                        if not data:
                            print("[FFMPEG] No more audio data (EOF)", flush=True)
                            break
                    buffer += data
                    bytes_received_total += len(data)

                    # Periodic debug logging (every 10 seconds)
                    if time.time() - last_debug_time > 10:
                        print(f"[DEBUG-TS-DATA] Total bytes received: {bytes_received_total}, buffer: {len(buffer)}", flush=True)
                        if self.backup_file and os.path.exists(self.backup_file):
                            ts_size = os.path.getsize(self.backup_file)
                            print(f"[DEBUG-TS-DATA] Backup file size: {ts_size} bytes ({ts_size/1024:.1f} KB)", flush=True)
                        else:
                            print(f"[DEBUG-TS-DATA] WARNING: Backup file not found: {self.backup_file}", flush=True)
                        last_debug_time = time.time()
                except BlockingIOError:
                    # No data available right now despite select saying ready
                    continue
                except OSError as e:
                    print(f"[FFMPEG] Read error: {e}", flush=True)
                    break

                # Process complete chunks from buffer
                while len(buffer) >= bytes_per_chunk:
                    audio_data = buffer[:bytes_per_chunk]
                    buffer = buffer[bytes_per_chunk:]

                    # If using queue mode (for compatibility with speech_recognition)
                    if self.data_queue is not None:
                        self.data_queue.put(audio_data)
                    else:
                        # Convert to numpy array
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        # Call the callback with the audio data
                        callback(audio_array)

                # Check if flush was requested (phrase timeout triggered)
                if self._flush_event.is_set():
                    self._flush_event.clear()
                    print(f"[DEBUG-TS-FLUSH] Flush triggered, buffer: {len(buffer)} bytes", flush=True)
                    if len(buffer) > 0 and self.data_queue is not None:
                        # Pad partial buffer with silence to make a full chunk
                        padding_needed = bytes_per_chunk - len(buffer)
                        padded_chunk = buffer + b'\x00' * padding_needed
                        original_size = len(buffer)
                        buffer = b''
                        self.data_queue.put(padded_chunk)
                        print(f"[FFMPEG] Flushed {original_size} bytes (padded with {padding_needed} bytes of silence)", flush=True)

        except Exception as e:
            print(f"[FFMPEG] Error in capture loop: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()

    def start(self, callback: Callable = None):
        """Start capturing audio

        Args:
            callback: Optional callback function. If None, uses queue mode (set data_queue first)
        """
        print(f"[DEBUG-TS-START] Starting audio capture", flush=True)
        print(f"[DEBUG-TS-START] Device: {self.device_name}, Sample rate: {self.sample_rate}", flush=True)

        # Reset file count for new session
        self._ts_file_count = 0

        if self.running:
            raise RuntimeError("Already capturing")

        # If no callback provided, we must have a queue
        if callback is None and self.data_queue is None:
            raise RuntimeError("Either callback or data_queue must be set")

        # Log existing backup files
        if self.backup_dir and os.path.exists(self.backup_dir):
            ts_files = [f for f in os.listdir(self.backup_dir) if f.endswith('.ts')]
            if ts_files:
                print(f"[AUDIO BACKUP] Found {len(ts_files)} existing backup files in {self.backup_dir}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, args=(callback if callback else lambda x: None,))
        self.thread.start()

        print(f"[DEBUG-TS-START] Capture thread started", flush=True)

        # Log the backup file location
        if self.backup_file:
            print(f"[AUDIO BACKUP] Recording to: {self.backup_file}")

    def stop(self):
        """Stop capturing audio"""
        print(f"[DEBUG-TS-STOP] Stopping audio capture", flush=True)
        if self.backup_file:
            if os.path.exists(self.backup_file):
                size = os.path.getsize(self.backup_file)
                print(f"[DEBUG-TS-STOP] Final backup file size: {size} bytes ({size/1024:.1f} KB)", flush=True)
            else:
                print(f"[DEBUG-TS-STOP] WARNING: Backup file missing: {self.backup_file}", flush=True)

        self.running = False

        # Terminate ffmpeg process first (before joining thread)
        if self.process:
            pid = self.process.pid
            print(f"[DEBUG-TS-STOP] Terminating ffmpeg process PID={pid}", flush=True)
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                    print(f"[DEBUG-TS-STOP] ffmpeg process terminated gracefully", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"[DEBUG-TS-STOP] ffmpeg not responding, sending SIGKILL", flush=True)
                    self.process.kill()
                    try:
                        self.process.wait(timeout=2)
                        print(f"[DEBUG-TS-STOP] ffmpeg process killed", flush=True)
                    except subprocess.TimeoutExpired:
                        # Last resort: use os.kill with SIGKILL
                        print(f"[DEBUG-TS-STOP] Force killing via os.kill({pid}, 9)", flush=True)
                        try:
                            os.kill(pid, 9)  # SIGKILL
                        except ProcessLookupError:
                            print(f"[DEBUG-TS-STOP] Process already dead", flush=True)
                        except Exception as e:
                            print(f"[DEBUG-TS-STOP] os.kill failed: {e}", flush=True)
            except Exception as e:
                print(f"[DEBUG-TS-STOP] Error terminating ffmpeg: {e}", flush=True)
            finally:
                self.process = None

        # Now join the capture thread
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                print(f"[DEBUG-TS-STOP] WARNING: Capture thread still alive after join timeout", flush=True)

        print(f"[DEBUG-TS-STOP] Audio capture stopped", flush=True)
        if self._ts_file_count > 1:
            print(f"[DEBUG-TS-SPLIT] *** SESSION HAD {self._ts_file_count} BACKUP FILES (recording was split {self._ts_file_count - 1} time(s)) ***", flush=True)

    @staticmethod
    def list_devices():
        """List available audio devices using ffmpeg"""
        import sys
        import os

        try:
            if sys.platform.startswith('linux'):
                devices = []

                # Method 1: Check /proc/asound/cards (most reliable)
                if os.path.exists('/proc/asound/cards'):
                    try:
                        with open('/proc/asound/cards', 'r') as f:
                            content = f.read()

                        # Parse card entries (format: " N [ID]: TYPE - NAME")
                        import re
                        # Match lines like " 0 [NVidia         ]: HDA-Intel - HDA NVidia"
                        card_pattern = r'^\s*(\d+)\s+\[([^\]]+)\]\s*:\s+([^-]+)\s*-\s*(.+)$'

                        for line in content.split('\n'):
                            match = re.match(card_pattern, line)
                            if match:
                                card_num = match.group(1).strip()
                                card_id = match.group(2).strip()
                                card_type = match.group(3).strip()
                                card_desc = match.group(4).strip()

                                # Use card description as the display name
                                display_name = f'{card_desc}'

                                # Use plughw instead of hw for better format compatibility
                                # plughw handles automatic format/rate conversion
                                devices.append({
                                    'name': f'plughw:{card_num},0',
                                    'index': len(devices),
                                    'display_name': display_name,
                                    'is_default': len(devices) == 0
                                })
                    except Exception as e:
                        print(f"[FFMPEG] Error reading /proc/asound/cards: {e}")

                # Method 2: Try arecord if available
                if not devices:
                    try:
                        cmd = ['arecord', '-L']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        for line in result.stdout.split('\n'):
                            line = line.strip()
                            if line and not line.startswith(' '):
                                devices.append({
                                    'name': line,
                                    'index': len(devices),
                                    'display_name': line,
                                    'is_default': len(devices) == 0
                                })
                    except FileNotFoundError:
                        pass  # arecord not available

                # Add default devices if nothing found
                if not devices:
                    devices.append({
                        'name': 'default',
                        'index': 0,
                        'display_name': 'Default Audio Device',
                        'is_default': True
                    })
                    devices.append({
                        'name': 'plughw:0,0',
                        'index': 1,
                        'display_name': 'Hardware Device 0',
                        'is_default': False
                    })

                return devices

            elif sys.platform == 'darwin':
                # List macOS devices
                cmd = ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                # Parse ffmpeg output for audio devices
                # Each audio device line looks like:
                #   [AVFoundation indev @ 0x...] [0] Built-in Microphone
                # Splitting on '] ' gives: ['[prefix', '[0', 'Built-in Microphone']
                devices = []
                in_audio_section = False
                for line in result.stderr.split('\n'):
                    if 'AVFoundation audio devices' in line:
                        in_audio_section = True
                        continue
                    if 'AVFoundation video devices' in line:
                        in_audio_section = False
                        continue
                    if in_audio_section and '] [' in line:
                        parts = line.split('] ')
                        if len(parts) >= 3:
                            # parts[1] is '[N' — extract the index
                            try:
                                idx = int(parts[1].lstrip('['))
                            except ValueError:
                                idx = len(devices)
                            # parts[2:] is the device name (rejoin in case name contained '] ')
                            display_name = '] '.join(parts[2:]).strip()
                            # The ffmpeg command builder prepends ':' to make ':N' for avfoundation
                            device_id = str(idx)
                            devices.append({
                                'name': device_id,
                                'display_name': display_name,
                                'index': idx,
                                'is_default': idx == 0,
                            })
                return devices

            elif sys.platform.startswith('win'):
                # List Windows devices
                cmd = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                devices = []
                for line in result.stderr.split('\n'):
                    if '"' in line and 'audio' in line.lower():
                        # Extract device name from quotes
                        start = line.find('"')
                        end = line.find('"', start + 1)
                        if start != -1 and end != -1:
                            name = line[start+1:end]
                            devices.append({'name': name, 'index': len(devices)})
                return devices

        except Exception as e:
            print(f"[FFMPEG] Error listing devices: {e}")
            return []


def list_audio_devices():
    """List available audio devices using ffmpeg"""
    return FFmpegAudioCapture.list_devices()


def create_compatible_audio_source(device_name=None, sample_rate=16000, backup_dir=None, filename_format=None, filename_prefix=None, ts_enabled=True):
    """
    Create an audio source compatible with the existing queue-based transcription system.

    Args:
        device_name: Audio device name (None for default)
        sample_rate: Sample rate in Hz (default 16000)
        backup_dir: Directory for MPEG-TS backup files (None for default _AUTOMATIC_BACKUP/YYYY/MM)
        filename_format: strftime format for backup filenames (None for default %Y-%m-%d_%H%M%S)
        filename_prefix: Optional prefix for backup filenames (None for no prefix)
        ts_enabled: Whether to enable .ts backup (default True)

    Returns:
        A source object with SAMPLE_WIDTH attribute and a data queue that can be used
        for audio capture using ffmpeg.
    """
    source = FFmpegAudioCapture(
        sample_rate=sample_rate,
        chunk_duration=1.0,
        device_name=device_name,
        backup_dir=backup_dir,
        filename_format=filename_format,
        filename_prefix=filename_prefix,
        ts_enabled=ts_enabled
    )
    source.data_queue = Queue()
    return source
