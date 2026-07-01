---
name: STT
description: Self-hosted real-time speech transcription platform for live event operators.
colors:
  bg-deep: "#050506"
  bg-base: "#0a0a0c"
  text-primary: "#EDEDEF"
  text-secondary: "#8A8F98"
  text-muted: "#5A5E66"
  accent: "#2979FF"
  accent-hover: "#5C9AFF"
  success: "#22C55E"
  warning: "#F59E0B"
  danger: "#EF4444"
  info: "#3B82F6"
typography:
  headline:
    fontFamily: "Inter, Segoe UI, -apple-system, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Inter, Segoe UI, -apple-system, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Inter, Segoe UI, -apple-system, sans-serif"
    fontSize: "0.88rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Inter, Segoe UI, -apple-system, sans-serif"
    fontSize: "0.82rem"
    fontWeight: 600
    lineHeight: 1.2
  caption:
    fontFamily: "Inter, Segoe UI, -apple-system, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.03em"
rounded:
  sm: "8px"
  md: "12px"
  lg: "16px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  xxl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "10px 20px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.sm}"
    padding: "8px 16px"
  button-danger:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "10px 20px"
  input-default:
    backgroundColor: "rgba(255,255,255,0.04)"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "10px 12px"
  nav-link:
    backgroundColor: "transparent"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
  nav-link-active:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
---

# Design System: STT

## 1. Overview

**Creative North Star: "The Mission Console"**

STT's control UI is an operator-grade control surface — the kind of interface that runs unattended through a two-hour church service or live event without the operator touching it again. Design decisions flow from that physical reality: someone under stage lighting, monitoring a feed they can't interrupt, trusting the tool to be right. Every pixel either carries a signal or earns its removal.

The visual identity is dark, dense, and direct. Near-black surfaces (`#050506`) eliminate glare in dim AV environments. A single accent drives active state and primary actions only — it appears at ≤10% of any surface, which makes its presence immediately meaningful. Borders define hierarchy; shadows appear only on modals. Type is Inter throughout at tighter scale ratios than a marketing page would use, because this is a control panel, not a showcase.

This system explicitly rejects the aesthetic it was born from: glassmorphism-by-default, indigo-violet gradients on every surface, floating glow cards, backdrop-blur on everything. Those are the 2024 AI dashboard template choices — recognizable on sight, trusted by nobody who knows what they're looking at. The control UI should read like it was designed by someone who understood the problem, not generated from the category.

**Scope note: The caption display (`index.html`) is a separate register.** It operates as a broadcast output — full-screen, large text, often composited in OBS or on a display wall. Its visual language (larger type, standalone dark frame, own color accent) is intentional and must not be modified by redesign work on the control UI. Two surfaces, two identities.

**Key Characteristics:**
- Near-black background; every surface tone is structural (page → panel → input), not decorative
- Single-family typography (Inter) at a compressed scale ratio; no display/body pairing
- Accent `#2979FF` (Electric Blue) at restraint; ≤10% of any surface
- Crisp 1px borders define containers; ambient shadows on modals only
- Status vocabulary (success / warning / danger / info) is the only multi-color system
- Motion: state transitions only, 150–200ms; no choreography during live sessions

## 2. Colors: The Operator Palette

A near-monochrome field punctuated by one accent and a strict four-color status vocabulary. The accent earns its color; everything else is a neutral.

### Primary
- **Electric Blue** (`#2979FF`): Accent. Active nav items, primary buttons, focus rings, and live-state indicators only. Used at ≤10% of any screen.
- **Electric Blue Lifted** (`#5C9AFF`): Hover state for accent elements only. Never used at rest.

### Neutral
- **Void** (`#050506`): Page background. The deepest surface. Nothing sits below this.
- **Station** (`#0a0a0c`): Panel and container background. One step above Void.
- **Ink** (`#EDEDEF`): Primary text. Near-white, not pure white — avoids harshness against near-black.
- **Dim** (`#8A8F98`): Secondary text, labels, descriptions. Pass WCAG AA against Void at ~5.1:1.
- **Ghost** (`#5A5E66`): Muted text, placeholders, disabled states. Use sparingly; verify contrast per use.

### Status Vocabulary
- **Live Green** (`#22C55E`): Success, active transcription, healthy state.
- **Caution Amber** (`#F59E0B`): Warnings, degraded state, models loading.
- **Alert Red** (`#EF4444`): Errors, denied segments, critical failure.
- **Signal Blue** (`#3B82F6`): Informational, neutral process state.

Each status color has a 10% opacity background variant for alert banners and badge fills. These are implementation-level values (`rgba(R,G,B,0.10)`), not design tokens.

### Named Rules
**The Restraint Rule.** Electric Blue appears on ≤10% of any given screen. Its scarcity is what makes it meaningful — the eye goes there because nothing else competes. The moment it decorates a card border, a section divider, or a hover animation, it has lost its function.

**The Electric Blue Standard.** The accent is `#2979FF` (Electric Blue). This replaced the prior indigo-violet `#7C6BF0`, which was a carry-over from the AI dashboard generation aesthetic. The colder, more technical hue reads as engineered infrastructure, not SaaS product. All surfaces use `#2979FF`.

**The Two-Surface Rule.** The caption display (`index.html`) owns its own palette (`#111111` background, `#bb86fc` accent, `#e0e0e0` text). Do not port control UI tokens into it, or vice versa.

## 3. Typography

**Body Font:** Inter (self-hosted, weights 300–700), fallback: `Segoe UI`, `-apple-system`, `BlinkMacSystemFont`, sans-serif.

**Character:** Single-family throughout. Product register — Inter at compressed scale ratios carries every typographic role without a display pairing. This is correct for a dense operator tool; a second font family would add visual noise, not hierarchy.

### Hierarchy
- **Headline** (600, 1.15rem, lh 1.3, ls −0.02em): Section page titles and major panel headers. One per view.
- **Title** (600, 0.95rem, lh 1.4, ls −0.01em): Card and sub-panel headers. The workhorse of this UI.
- **Body** (400, 0.88rem, lh 1.5): Descriptions, help text, setting explanations. Line length cap at 65ch in prose contexts.
- **Label** (600, 0.82rem, lh 1.2): Form labels, nav items, button text, table headers. The most common type role.
- **Caption** (700, 0.72rem, lh 1.2, ls +0.03em): Status badges, mode indicators, tiny metadata. Uppercase only for `LIVE`, `PANNs`, `ENERGY` badges — not as section kickers.

### Named Rules
**The Mono Exception.** Numeric values emitted by the system (audio level dB, timestamps, confidence scores, version numbers) render in `font-variant-numeric: tabular-nums` and optionally in a monospace stack when alignment precision matters. This is the only permitted departure from Inter for data.

**The Single-Weight Heading Rule.** All headings use weight 600. No display weight 800+ in the control UI — that weight belongs to the caption display output screen, not the settings panel.

## 4. Elevation

Depth is expressed through tonal layering and crisp 1px borders. Shadows are reserved for one job: floating overlays that must visually detach from the page.

**Surface stack (no shadows needed):**
- Page: `#050506` (Void)
- Panels / content areas: `#0a0a0c` (Station)
- Cards / form groups: `rgba(255,255,255,0.05)` tinted above Station
- Inputs: `rgba(255,255,255,0.04)`

**Border vocabulary:**
- Default container border: `1px solid rgba(255,255,255,0.08)`
- Hover state: `1px solid rgba(255,255,255,0.15)`
- Focus ring: `0 0 0 3px rgba(41,121,255,0.2)` (accent-glow)

### Shadow Vocabulary
- **Modal lift** (`0 24px 80px rgba(0,0,0,0.6)`): Dialog and modal overlays only. Establishes full detachment from the page layer.
- **Accent hover** (`0 4px 16px rgba(41,121,255,0.25)`): Primary button on hover. The only decorative shadow permitted; it grounds the elevated button.

### Named Rules
**The Border-First Rule.** Surfaces are defined by borders, not shadows. If you're reaching for a `box-shadow` to define a container, reach for a border instead. Shadows exist for modal lift and single accent-button hover — that is the complete list.

**The No-Backdrop-Default Rule.** `backdrop-filter: blur()` is permitted on the nav bar (established pattern) and modal overlays. It is prohibited as a card treatment, a hover effect, or a decoration on settings panels. Glassmorphism-by-default is one of this system's explicit anti-references.

## 5. Components

### Buttons

Compact, border-radius 8px (`--radius-sm`), font-weight 600, 0.82rem. Transitions 200ms on background, color, transform, and box-shadow.

- **Primary:** `background: #2979FF`, `color: #EDEDEF`, padding `10px 20px`. Hover: translateY(-1px), accent-glow shadow. Active: scale(0.98).
- **Ghost:** `background: transparent`, `color: #8A8F98`, `border: 1px solid rgba(255,255,255,0.12)`. Hover: `color: #EDEDEF`, border lifts to `rgba(255,255,255,0.20)`. No shadow.
- **Danger:** `background: #EF4444`, `color: #EDEDEF`. Hover: `background: #f87171`. Used for destructive actions only.
- **Icon-only:** 36px square, ghost style, centered SVG icon. No label. Requires `aria-label`.

### Inputs / Fields

Single-line: `background: rgba(255,255,255,0.04)`, `border: 1px solid rgba(255,255,255,0.08)`, `border-radius: 8px`, padding `10px 12px`, font-size 0.82rem. Focus: border lifts to `rgba(255,255,255,0.20)`, focus ring `0 0 0 3px rgba(41,121,255,0.2)`. Error state: border `#EF4444`, focus ring red-tinted.

Range sliders follow the same style contract: thumb uses accent color, track uses `rgba(255,255,255,0.1)` fill.

### Cards / Panels

- **Corner style:** 12px (`--radius-md`)
- **Background:** `rgba(255,255,255,0.05)` over Station
- **Border:** `1px solid rgba(255,255,255,0.08)`
- **Shadow:** none at rest; modal lift only for overlay panels
- **Internal padding:** 16px standard, 12px compact (dense settings groups)

No side-stripe accent borders. If a card needs emphasis, use a full-border color change or a tinted background, not a colored left edge.

### Navigation

Glassmorphic header bar: `background: rgba(10,10,12,0.8)`, `backdrop-filter: blur(12px)`, bottom border `rgba(255,255,255,0.08)`. Sticky.

- **Nav link (default):** Label style, `color: #8A8F98`, padding `6px 12px`, radius 8px. Hover: `color: #EDEDEF`, `background: rgba(255,255,255,0.06)`.
- **Nav link (active):** `background: #2979FF`, `color: #EDEDEF`, `box-shadow: 0 0 16px rgba(41,121,255,0.3)`. The only place accent appears in the nav.
- **Mobile:** collapses to hamburger; nav panel drops full-width below header.

### Status Badges (Signature Component)

The PANNs audio classifier emits three states — Speaking, Music, Quiet — plus a detector-mode badge (PANNs / Energy). These are the most live, most operator-critical elements in the UI.

- **Speaking:** `color: #22C55E` (Live Green), font-weight 600
- **Music:** `color: #2979FF` (Electric Blue), font-weight 600
- **Quiet:** `color: #5A5E66` (Ghost), font-weight 600
- **PANNs badge:** caption style, `color: #2979FF`, `background: rgba(41,121,255,0.15)`, padding `1px 6px`, radius 8px
- **Energy badge:** caption style, `color: #dd6b20`, `background: rgba(221,107,32,0.15)`, padding `1px 6px`, radius 8px
- **Advanced label** (`#audio_tag_display`): CNN14 class name in Ghost (`#5A5E66`), music% in accent color. Appears only when PANNs is active.

### Toggle / Checkbox

Custom toggle: 40×22px pill. Inactive: `background: rgba(255,255,255,0.1)`. Active: `background: #2979FF`. Thumb: white circle, transition 0.2s ease-out. Focus ring on the parent label.

## 6. Do's and Don'ts

### Do:
- **Do** use `#2979FF` only for primary actions, active/selected state, and focus indicators. It should be the rarest saturated color on any screen.
- **Do** define container depth through `1px solid rgba(255,255,255,0.08)` borders and tonal surface steps. That is the elevation system.
- **Do** use the status vocabulary (green / amber / red / blue) as the only multi-color system. Every other informational color routes through one of these four.
- **Do** use `font-variant-numeric: tabular-nums` on all numeric data outputs (dB, confidence, timing values). Columns must align.
- **Do** respect the two-surface rule: control UI and caption display are separate design contexts. Never cross-port tokens.
- **Do** keep motion to state transitions: 150–200ms, `cubic-bezier(0.16, 1, 0.3, 1)` ease-out, on background/color/transform only. Add `@media (prefers-reduced-motion: reduce)` alternatives; operators running live events cannot tolerate motion surprises.
- **Do** treat WCAG AA as a floor, not a target. High-contrast readability under stage lighting or bright monitor glare is the actual constraint.
- **Do** use Electric Blue `#2979FF` as the single accent across all control UI surfaces.

### Don't:
- **Don't** use `backdrop-filter: blur()` on cards, settings panels, or hover effects. It's permitted on the nav bar and modal overlays — nowhere else. Glassmorphism-by-default is an explicit anti-reference for this system.
- **Don't** use `border-left` greater than 1px as a colored accent stripe on cards or list items. If you're reaching for this, use a full tinted background or a leading icon instead.
- **Don't** use `background-clip: text` with a gradient (gradient text). Single solid color for all text; emphasis through weight or size.
- **Don't** put Electric Blue on more than ~10% of any screen surface. Rarity is its function.
- **Don't** animate layout properties (width, height, margin, padding). Transition transform, opacity, color, and box-shadow only.
- **Don't** design the control UI to look like a consumer transcription app (Otter.ai, Rev, Descript). Rounded, friendly, spacious — those are all wrong registers. Dense, direct, operator-grade.
- **Don't** let the design read as "AI generated this." Cross-check: if the color + layout could have come from a dark-mode SaaS template, replace at least one element with a deliberate choice. The current glassmorphism card system is the primary offender.
- **Don't** add motion to the live transcription view or audio monitoring UI. When an operator is watching a live feed, no animation should compete with the content.
- **Don't** touch `index.html` (caption display) in control UI redesign work. Different surface, different register, different constraints.
