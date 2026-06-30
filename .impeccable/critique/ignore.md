# Critique ignore list

## design-system-color: progress track backgrounds

- `rgba(255,255,255,0.1)` — disk space bar track background in file-manager; intentional thin-track contrast value

## design-system-color: semantic overlay and themed panel colors (translation.html)

- `rgba(26, 26, 46, 0.95)` — unsaved-bar frosted dark backdrop; intentional semi-opaque blue-tinted overlay
- `rgba(224, 224, 224, 0.6)` — in-progress text in translation preview (same semantic as the 0.7 variant in live-settings); intentional reduced opacity for in-progress state
- `#1a2a3a`, `#3a6a9a` — pairing request banner blue theme (blue = connection/network semantic)
- `#1e2a1e`, `#3a5a3a` — pair code entry green theme (green = success/confirmation semantic)
- `#f0e040` — pairing code display high-visibility yellow; operator needs to read it quickly
- `#7ab8f5` — pairing IP address accent blue in banner; informational blue
- `#2a2000`, `#665500`, `#ffcc00` — Whisper Translate warning yellow theme (yellow = caution semantic)
- `#2a1500`, `#664400`, `#ff9900` — Whisper Forced Language warning orange theme (orange = experimental caution)

## design-system-color: intentional hover shades

- `#16A34A` — success button hover shade (matches base.html `.btn-success:hover`); intentional system hover
- `#DC2626` — danger button hover shade (matches base.html `.btn-danger:hover`); intentional system hover
- `rgba(224, 224, 224, 0.7)` — in-progress transcription text at reduced opacity; intentional for live status distinction
- `#38a169` — model-btn.download hover shade (darker success green); intentional hover
- `#e53e3e` — model-btn.remove hover shade (darker danger red); intentional hover
- `#3182ce` — file-manager download button hover shade (darker info blue); intentional hover
- `#718096` — file-manager up-button hover shade (darker neutral); intentional for navigation affordance

## design-system-radius: intentional pill/micro shapes

- `3px` — disk space bar track; intentional micro-pill for thin progress track
- `4px` — compact UI elements: file-manager action buttons, file mover textareas, translation language picker items, dense inline form inputs; intentional micro-radius for tight/dense contexts
- `10px` — progress bar container pills (upload, download, PANNs progress); intentional pill shape
- `12px` — status badge pills ("✓ Downloaded", "Available"); intentional badge pill

## design-system-font: deferred

- `Courier New` (line ~234 in live-settings.html) — `.preview-content` monospace; deferred to typeset pass

## design-system-font: caption display configuration values

- `Arial` (fontFamily default in url-builder.html) — this is a user-configurable caption font setting passed as a URL parameter to the caption display (index.html). Not a UI font; users can change it in the builder. Ignore as a UI font violation.

## em-dash-overuse: false positive from Jinja template blocks

- CSS `--custom-property` names in `{% block extra_css %}` blocks are not wrapped in `<style>` tags, so the HTML stripper leaves them as text. Each `--accent`, `--border` etc. matches the `--(?=\S)` regex, inflating em-dash counts in all Jinja-extended templates. Actual em-dash usage is ≤2 per page.
