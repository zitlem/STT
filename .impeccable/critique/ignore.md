# Critique ignore list

## design-system-color: intentional hover shades

- `#16A34A` — success button hover shade (matches base.html `.btn-success:hover`); intentional system hover
- `#DC2626` — danger button hover shade (matches base.html `.btn-danger:hover`); intentional system hover
- `rgba(224, 224, 224, 0.7)` — in-progress transcription text at reduced opacity; intentional for live status distinction
- `#38a169` — model-btn.download hover shade (darker success green); intentional hover
- `#e53e3e` — model-btn.remove hover shade (darker danger red); intentional hover

## design-system-radius: intentional pill/micro shapes

- `3px` — disk space bar track; intentional micro-pill for thin progress track
- `10px` — progress bar container pills (upload, download, PANNs progress); intentional pill shape
- `12px` — status badge pills ("✓ Downloaded", "Available"); intentional badge pill

## design-system-font: deferred

- `Courier New` (line ~234 in live-settings.html) — `.preview-content` monospace; deferred to typeset pass

## em-dash-overuse: false positive from Jinja template blocks

- CSS `--custom-property` names in `{% block extra_css %}` blocks are not wrapped in `<style>` tags, so the HTML stripper leaves them as text. Each `--accent`, `--border` etc. matches the `--(?=\S)` regex, inflating em-dash counts in all Jinja-extended templates. Actual em-dash usage is ≤2 per page.
