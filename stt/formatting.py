"""Transcript formatting and export helpers (SRT/VTT/HTML, sizes).

Extracted from speech_to_text.py so they can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only. Anything that
needs runtime config takes it as an explicit parameter (html_enabled,
highlight_config_path); thin wrappers in speech_to_text.py supply the live
values.
"""

import html
import json
import os
import re
import sqlite3

def format_transcription(segments, format_type):
    """
    Format transcription segments into requested format.

    Args:
        segments: List of (text, start_time, end_time) tuples
        format_type: One of 'txt', 'srt', 'vtt', 'json'

    Returns:
        Formatted transcription string
    """
    if format_type == "txt":
        # Plain text - just concatenate all text
        return "\n".join([seg["text"] for seg in segments])

    elif format_type == "srt":
        # SubRip format
        output = []
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg["start"])
            end = format_timestamp_srt(seg["end"])
            output.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
        return "\n".join(output)

    elif format_type == "vtt":
        # WebVTT format
        output = ["WEBVTT\n"]
        for seg in segments:
            start = format_timestamp_vtt(seg["start"])
            end = format_timestamp_vtt(seg["end"])
            output.append(f"{start} --> {end}\n{seg['text']}\n")
        return "\n".join(output)

    elif format_type == "json":
        # JSON format with metadata
        return json.dumps(
            {
                "segments": segments,
                "total_segments": len(segments),
                "duration": segments[-1]["end"] if segments else 0,
            },
            indent=2,
        )

    else:
        return "\n".join([seg["text"] for seg in segments])


def format_timestamp_srt(seconds):
    """Format timestamp for SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds):
    """Format timestamp for VTT format: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def convert_db_to_srt(db_path, html_enabled=True, highlight_config_path=None):
    """
    Convert a Transcriptions.db file to SRT subtitle format.

    Args:
        db_path: Path to the .db file
        html_enabled: also generate the HTML export alongside the SRT
        highlight_config_path: word_highlighting.json path for the HTML export

    Returns:
        Path to the created .srt file, or None on failure
    """
    from datetime import datetime

    try:
        if not db_path or not os.path.exists(db_path):
            print(f"[SRT] Database not found: {db_path}")
            return None

        # Read entries from database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, text FROM transcriptions
                WHERE timestamp IS NOT NULL AND timestamp != ''
                AND text IS NOT NULL AND TRIM(text) != '' AND TRIM(text) != ' '
                AND COALESCE(denied, 0) = 0
                AND COALESCE(is_final, 1) = 1
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print("[SRT] No valid entries found in database")
            return None

        # Parse timestamps and calculate durations
        segments = []
        first_time = None

        for i, (timestamp_str, text) in enumerate(entries):
            try:
                # Parse ISO timestamp (handles both "2025-01-02 12:34:56" and "2025-01-02T12:34:56")
                ts_normalized = timestamp_str.replace("T", " ").strip()
                dt = datetime.strptime(ts_normalized, "%Y-%m-%d %H:%M:%S")

                if first_time is None:
                    first_time = dt

                # Calculate seconds from start
                start_seconds = (dt - first_time).total_seconds()

                # Estimate end time (use next entry's start or add 3 seconds)
                if i + 1 < len(entries):
                    next_ts = entries[i + 1][0].replace("T", " ").strip()
                    next_dt = datetime.strptime(next_ts, "%Y-%m-%d %H:%M:%S")
                    end_seconds = (next_dt - first_time).total_seconds()
                    # Ensure minimum 1 second duration
                    if end_seconds <= start_seconds:
                        end_seconds = start_seconds + 1.0
                else:
                    end_seconds = start_seconds + 3.0

                segments.append(
                    {"text": text.strip(), "start": start_seconds, "end": end_seconds}
                )
            except Exception as e:
                print(f"[SRT] Error parsing entry {i}: {e}")
                continue

        if not segments:
            print("[SRT] No valid segments after parsing")
            return None

        # Format as SRT using existing function
        srt_content = format_transcription(segments, "srt")

        # Write SRT file alongside the database
        srt_path = db_path.replace(".db", ".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"[SRT] Created: {srt_path} ({len(segments)} entries)")

        # Also generate HTML with word highlighting (if enabled)
        if html_enabled:
            try:
                convert_db_to_html(db_path, highlight_config_path)
            except Exception as e:
                print(f"[SRT] Failed to generate HTML: {e}")
        else:
            print("[HTML] HTML generation disabled in settings")

        return srt_path
    except Exception as e:
        print(f"[SRT] Error converting database to SRT: {e}")
        import traceback

        traceback.print_exc()
        return None

def convert_db_to_translation_srt(db_path):
    """Convert translated text from a Transcriptions.db to SRT subtitle format."""
    from datetime import datetime

    try:
        if not db_path or not os.path.exists(db_path):
            return None

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, translated_text FROM transcriptions
                WHERE timestamp IS NOT NULL AND timestamp != ''
                AND translated_text IS NOT NULL AND TRIM(translated_text) != ''
                AND COALESCE(denied, 0) = 0
                AND COALESCE(is_final, 1) = 1
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print("[SRT-TRANSLATION] No translated entries found in database")
            return None

        segments = []
        first_time = None

        for i, (timestamp_str, text) in enumerate(entries):
            try:
                ts_normalized = timestamp_str.replace("T", " ").strip()
                dt = datetime.strptime(ts_normalized, "%Y-%m-%d %H:%M:%S")

                if first_time is None:
                    first_time = dt

                start_seconds = (dt - first_time).total_seconds()

                if i + 1 < len(entries):
                    next_ts = entries[i + 1][0].replace("T", " ").strip()
                    next_dt = datetime.strptime(next_ts, "%Y-%m-%d %H:%M:%S")
                    end_seconds = (next_dt - first_time).total_seconds()
                    if end_seconds <= start_seconds:
                        end_seconds = start_seconds + 1.0
                else:
                    end_seconds = start_seconds + 3.0

                segments.append(
                    {"text": text.strip(), "start": start_seconds, "end": end_seconds}
                )
            except Exception:
                continue

        if not segments:
            return None

        srt_content = format_transcription(segments, "srt")
        srt_path = db_path.replace(".db", ".translated.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"[SRT-TRANSLATION] Created: {srt_path} ({len(segments)} entries)")
        return srt_path
    except Exception as e:
        print(f"[SRT-TRANSLATION] Error: {e}")
        return None

def apply_word_highlighting_server(text, config):
    """
    Apply word highlighting to text using the highlighting configuration.
    Server-side equivalent of the JavaScript applyWordHighlighting() function.

    Args:
        text: The text to apply highlighting to
        config: The word highlighting configuration dict

    Returns:
        Text with HTML span tags for highlighting
    """
    if not config or not config.get("enabled") or not config.get("words"):
        return html.escape(text)

    # First escape HTML entities in the original text
    result = html.escape(text)

    # Get list of disabled color groups
    disabled_colors = config.get("disabled_colors", [])

    for word_entry in config["words"]:
        word = word_entry.get("word", "")
        color = word_entry.get("color", "#ffff00")
        is_regex = word_entry.get("is_regex", False)
        case_sensitive = word_entry.get("case_sensitive", False)

        if not word:
            continue

        # Skip if this color group is disabled
        if color in disabled_colors:
            continue

        try:
            flags = 0 if case_sensitive else re.IGNORECASE

            if is_regex:
                # Use the pattern as-is for regex mode
                # Use word boundaries that work with Unicode
                pattern = r"(?<![A-Za-z\u0400-\u04FF0-9])(" + word + r")(?![A-Za-z\u0400-\u04FF0-9])"
            else:
                # Escape special regex characters for plain words
                escaped_word = re.escape(word)
                pattern = r"(?<![A-Za-z\u0400-\u04FF0-9])(" + escaped_word + r")(?![A-Za-z\u0400-\u04FF0-9])"

            # Replace matches with highlighted spans
            result = re.sub(
                pattern,
                f'<span style="color: {color};">\\1</span>',
                result,
                flags=flags,
            )
        except re.error as e:
            print(f"[HTML] Invalid regex pattern '{word}': {e}")
            continue

    return result

def convert_db_to_html(db_path, highlight_config_path=None):
    """
    Convert a Transcriptions.db file to HTML format with word highlighting.

    Args:
        db_path: Path to the .db file
        highlight_config_path: path to word_highlighting.json (optional)

    Returns:
        Path to the created .html file, or None on failure
    """
    from datetime import datetime

    try:
        if not db_path or not os.path.exists(db_path):
            print(f"[HTML] Database not found: {db_path}")
            return None

        # Load word highlighting configuration
        highlight_config = None
        if highlight_config_path and os.path.exists(highlight_config_path):
            try:
                with open(highlight_config_path, "r", encoding="utf-8") as f:
                    highlight_config = json.load(f)
            except Exception as e:
                print(f"[HTML] Error loading word highlighting config: {e}")

        # Read entries from database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Older .db files predate the 'marked' column and COALESCE can't
            # save a missing column, so probe the schema first.
            cursor.execute("PRAGMA table_info(transcriptions)")
            has_marked = any(row[1] == "marked" for row in cursor.fetchall())
            marked_col = "COALESCE(marked, 0)" if has_marked else "0"
            cursor.execute(
                f"""
                SELECT timestamp, text, {marked_col} FROM transcriptions
                WHERE timestamp IS NOT NULL AND timestamp != ''
                AND text IS NOT NULL AND TRIM(text) != '' AND TRIM(text) != ' '
                AND COALESCE(denied, 0) = 0
                AND COALESCE(is_final, 1) = 1
                ORDER BY id ASC
            """
            )
            entries = cursor.fetchall()

        if not entries:
            print("[HTML] No valid entries found in database")
            return None

        # Parse timestamps and build segments
        segments_html = []
        first_time = None

        for i, (timestamp_str, text, row_marked) in enumerate(entries):
            try:
                # Parse ISO timestamp
                ts_normalized = timestamp_str.replace("T", " ").strip()
                dt = datetime.strptime(ts_normalized, "%Y-%m-%d %H:%M:%S")

                if first_time is None:
                    first_time = dt

                # Calculate both clock time and elapsed time
                clock_time = dt.strftime("%H:%M:%S")
                elapsed = dt - first_time
                elapsed_hours = int(elapsed.total_seconds() // 3600)
                elapsed_minutes = int((elapsed.total_seconds() % 3600) // 60)
                elapsed_seconds = int(elapsed.total_seconds() % 60)
                elapsed_time = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"

                # Apply word highlighting
                highlighted_text = apply_word_highlighting_server(text.strip(), highlight_config)

                seg_cls = "segment marked" if row_marked else "segment"
                mark_badge = '<span class="mark-badge" title="Marked during session">&#9873;</span>' if row_marked else ""
                segments_html.append(
                    f'<div class="{seg_cls}"><span class="timestamp" data-clock="{clock_time}" data-elapsed="{elapsed_time}">[{clock_time}]</span><span class="text">{highlighted_text}</span>{mark_badge}</div>'
                )
            except Exception as e:
                print(f"[HTML] Error parsing entry {i}: {e}")
                continue

        if not segments_html:
            print("[HTML] No valid segments after parsing")
            return None

        # Get the date for the title
        title_date = first_time.strftime("%Y-%m-%d %H:%M") if first_time else "Unknown"

        # Build the HTML document
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcription - {title_date}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #252525;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #bb86fc;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #bb86fc;
        }}
        .meta {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 25px;
        }}
        .segment {{
            margin: 12px 0;
            padding: 10px 15px;
            background: #2a2a2a;
            border-radius: 6px;
            border-left: 3px solid #3a3a3a;
        }}
        .segment:hover {{
            border-left-color: #bb86fc;
        }}
        .segment.marked {{
            border-left-color: #ffb74d;
            background: #2e2a22;
        }}
        .mark-badge {{
            color: #ffb74d;
            margin-left: 10px;
            font-size: 0.9em;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.8em;
            margin-right: 12px;
            font-family: monospace;
        }}
        /* Controls panel */
        .controls {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: #2a2a2a;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}
        .controls-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            width: 100%;
        }}
        .controls label {{
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .controls input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .controls input[type="text"] {{
            background: #1a1a1a;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 8px 12px;
            color: #e0e0e0;
            font-size: 0.9em;
            width: 200px;
        }}
        .controls input[type="text"]:focus {{
            outline: none;
            border-color: #bb86fc;
        }}
        .controls button {{
            background: #bb86fc;
            color: #1a1a1a;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }}
        .controls button:hover {{
            background: #9a67ea;
        }}
        .controls button.secondary {{
            background: #3a3a3a;
            color: #e0e0e0;
        }}
        .controls button.secondary:hover {{
            background: #4a4a4a;
        }}
        .font-controls {{
            display: flex;
            gap: 5px;
            align-items: center;
        }}
        .font-controls button {{
            width: 32px;
            height: 32px;
            padding: 0;
            font-size: 1.1em;
        }}
        .font-controls span {{
            font-size: 0.85em;
            color: #888;
            min-width: 40px;
            text-align: center;
        }}
        .search-count {{
            font-size: 0.85em;
            color: #888;
            margin-left: 8px;
        }}
        .search-highlight {{
            background: #ffeb3b !important;
            color: #000 !important;
            padding: 1px 2px;
            border-radius: 2px;
        }}
        .search-highlight-active {{
            background: #ff6d00 !important;
            color: #fff !important;
            padding: 1px 2px;
            border-radius: 2px;
        }}
        .controls-spacer {{
            flex: 1;
        }}
        /* Printer mode - white background */
        body.printer-mode {{
            background: #fff;
            color: #000;
        }}
        body.printer-mode .container {{
            background: #fff;
            box-shadow: none;
        }}
        body.printer-mode .segment {{
            background: #f5f5f5;
            border-left-color: #ccc;
        }}
        body.printer-mode h1 {{
            color: #333;
            border-bottom-color: #333;
        }}
        body.printer-mode .meta {{
            color: #666;
        }}
        body.printer-mode .timestamp {{
            color: #666;
        }}
        body.printer-mode .controls {{
            background: #f0f0f0;
        }}
        body.printer-mode .controls input[type="text"] {{
            background: #fff;
            border-color: #ccc;
            color: #000;
        }}
        body.printer-mode .controls button.secondary {{
            background: #e0e0e0;
            color: #333;
        }}
        /* Hide timestamps */
        body.hide-timestamps .timestamp {{
            display: none;
        }}
        /* No rows - continuous text */
        body.no-rows .segment {{
            display: inline;
            background: none;
            border: none;
            padding: 0;
            margin: 0;
            border-radius: 0;
        }}
        body.no-rows .segment::after {{
            content: ' ';
        }}
        /* Hide word highlighting */
        body.no-highlighting span[style*="color:"] {{
            color: inherit !important;
        }}
        /* Print styles */
        @media print {{
            .no-print {{
                display: none !important;
            }}
            body {{
                background: #fff !important;
                color: #000 !important;
                padding: 0 !important;
            }}
            .container {{
                background: #fff !important;
                box-shadow: none !important;
                padding: 0 !important;
            }}
            .segment {{
                background: #fff !important;
                border: none !important;
                padding: 5px 0 !important;
                margin: 5px 0 !important;
            }}
            h1 {{
                color: #000 !important;
                border-bottom-color: #000 !important;
            }}
            .meta {{
                color: #333 !important;
            }}
            .timestamp {{
                color: #333 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Transcription</h1>
        <div class="controls no-print">
            <div class="controls-row">
                <label><input type="checkbox" id="showTimestamps" checked> Timestamps</label>
                <label><input type="checkbox" id="useElapsed"> Elapsed Time</label>
                <label><input type="checkbox" id="showRows" checked> Row Separators</label>
                <label><input type="checkbox" id="printerMode"> Printer Mode</label>
                <label><input type="checkbox" id="showHighlighting" checked> Highlighting</label>
                <div class="font-controls">
                    <button class="secondary" onclick="changeFontSize(-1)">A-</button>
                    <span id="fontSizeDisplay">100%</span>
                    <button class="secondary" onclick="changeFontSize(1)">A+</button>
                </div>
                <div class="controls-spacer"></div>
                <button class="secondary" onclick="copyToClipboard()">Copy Text</button>
                <button onclick="window.print()">Print</button>
            </div>
            <div class="controls-row">
                <input type="text" id="searchInput" placeholder="Search text..." onkeyup="searchText(event)">
                <button class="secondary" onclick="doSearch()">Search</button>
                <button class="secondary" onclick="prevMatch()">&uarr; Prev</button>
                <button class="secondary" onclick="nextMatch()">&darr; Next</button>
                <span id="searchCount" class="search-count"></span>
                <button class="secondary" onclick="clearSearch()">Clear</button>
            </div>
        </div>
        <div class="meta">
            <p>Date: {title_date}</p>
            <p>Segments: {len(segments_html)}</p>
        </div>
        {"".join(segments_html)}
    </div>
    <script>
        // Toggle controls
        document.getElementById('showTimestamps').addEventListener('change', function() {{
            document.body.classList.toggle('hide-timestamps', !this.checked);
        }});
        document.getElementById('useElapsed').addEventListener('change', function() {{
            const useElapsed = this.checked;
            document.querySelectorAll('.timestamp').forEach(ts => {{
                const value = useElapsed ? ts.dataset.elapsed : ts.dataset.clock;
                ts.textContent = '[' + value + ']';
            }});
        }});
        document.getElementById('showRows').addEventListener('change', function() {{
            document.body.classList.toggle('no-rows', !this.checked);
        }});
        document.getElementById('printerMode').addEventListener('change', function() {{
            document.body.classList.toggle('printer-mode', this.checked);
        }});
        document.getElementById('showHighlighting').addEventListener('change', function() {{
            document.body.classList.toggle('no-highlighting', !this.checked);
        }});

        // Font size controls
        let currentFontSize = 100;
        function changeFontSize(delta) {{
            currentFontSize = Math.max(50, Math.min(200, currentFontSize + delta * 10));
            document.querySelector('.container').style.fontSize = currentFontSize + '%';
            document.getElementById('fontSizeDisplay').textContent = currentFontSize + '%';
        }}

        // Copy to clipboard
        function copyToClipboard() {{
            const segments = document.querySelectorAll('.segment');
            const showTimestamps = document.getElementById('showTimestamps').checked;
            let text = '';
            segments.forEach(seg => {{
                if (showTimestamps) {{
                    const ts = seg.querySelector('.timestamp');
                    if (ts) text += ts.textContent + ' ';
                }}
                // Get text content without HTML tags
                const clone = seg.cloneNode(true);
                const tsClone = clone.querySelector('.timestamp');
                if (tsClone) tsClone.remove();
                text += clone.textContent.trim() + '\\n';
            }});
            navigator.clipboard.writeText(text.trim()).then(() => {{
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = originalText, 1500);
            }});
        }}

        // Search functionality
        let originalContent = {{}};
        let searchMatches = [];
        let currentMatchIndex = -1;

        function searchText(event) {{
            if (event.key === 'Enter') doSearch();
            else if (event.key === 'ArrowDown') {{ event.preventDefault(); nextMatch(); }}
            else if (event.key === 'ArrowUp') {{ event.preventDefault(); prevMatch(); }}
        }}
        function doSearch() {{
            const query = document.getElementById('searchInput').value.trim().toLowerCase();
            clearSearch();
            if (!query) return;

            const segments = document.querySelectorAll('.segment');
            const regex = new RegExp('(' + query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
            segments.forEach((seg, idx) => {{
                if (!originalContent[idx]) {{
                    originalContent[idx] = seg.innerHTML;
                }}
                if (seg.textContent.toLowerCase().includes(query)) {{
                    seg.innerHTML = seg.innerHTML.replace(/>([^<]+)</g, (match, content) => {{
                        const highlighted = content.replace(regex, '<span class="search-highlight">$1</span>');
                        return '>' + highlighted + '<';
                    }});
                }}
            }});
            searchMatches = Array.from(document.querySelectorAll('.search-highlight'));
            currentMatchIndex = searchMatches.length > 0 ? 0 : -1;
            updateActiveMatch();
        }}
        function updateActiveMatch() {{
            searchMatches.forEach((el, i) => {{
                el.className = i === currentMatchIndex ? 'search-highlight-active' : 'search-highlight';
            }});
            const counter = document.getElementById('searchCount');
            if (searchMatches.length === 0) {{
                counter.textContent = 'No matches';
            }} else {{
                counter.textContent = (currentMatchIndex + 1) + ' / ' + searchMatches.length;
                searchMatches[currentMatchIndex].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}
        function nextMatch() {{
            if (searchMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex + 1) % searchMatches.length;
            updateActiveMatch();
        }}
        function prevMatch() {{
            if (searchMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
            updateActiveMatch();
        }}
        function clearSearch() {{
            const segments = document.querySelectorAll('.segment');
            segments.forEach((seg, idx) => {{
                if (originalContent[idx]) {{
                    seg.innerHTML = originalContent[idx];
                }}
            }});
            originalContent = {{}};
            searchMatches = [];
            currentMatchIndex = -1;
            document.getElementById('searchCount').textContent = '';
        }}
    </script>
</body>
</html>"""

        # Write HTML file alongside the database
        html_path = db_path.replace(".db", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[HTML] Created: {html_path} ({len(segments_html)} entries)")
        return html_path
    except Exception as e:
        print(f"[HTML] Error converting database to HTML: {e}")
        import traceback

        traceback.print_exc()
        return None

def format_file_size(bytes_value):
    """Convert bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"
