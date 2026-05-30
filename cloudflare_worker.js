/**
 * STT Crash Reporter — Cloudflare Worker
 *
 * Receives sanitized crash reports from watchdog.py and files GitHub issues.
 * The GitHub token never leaves Cloudflare — users only need to set
 * crash_reporting.enabled = true in their config.json.
 *
 * ── One-time setup ─────────────────────────────────────────────────────────
 *
 *  1. Install Wrangler and log in:
 *       npm install -g wrangler
 *       wrangler login
 *
 *  2. (Recommended) Create a KV namespace for server-side rate limiting:
 *       wrangler kv:namespace create CRASH_CACHE
 *     Copy the returned id into wrangler.toml (see the commented block).
 *
 *  3. Add secrets (never committed to git):
 *       wrangler secret put GITHUB_TOKEN
 *         → Fine-grained PAT: Issues → Read & write on zitlem/STT only
 *       wrangler secret put CRASH_API_KEY
 *         → Must match _CRASH_API_KEY in watchdog.py
 *
 *  4. Deploy:
 *       wrangler deploy
 *
 *  5. Copy the printed Worker URL into watchdog.py → _CRASH_WORKER_URL
 *     then commit and tag a new release.
 *
 * ───────────────────────────────────────────────────────────────────────────
 */

const GITHUB_REPO  = 'zitlem/STT';
const MAX_BODY     = 65_536;   // 64 KB hard cap on incoming payload
const KV_COOLDOWN  = 600;      // server-side rate limit per fingerprint (seconds)

export default {
  async fetch(request, env) {

    if (request.method !== 'POST') {
      return reply({ error: 'Method not allowed' }, 405);
    }

    // ── Authenticate ─────────────────────────────────────────────────────
    const apiKey = request.headers.get('X-Api-Key') ?? '';
    if (!env.CRASH_API_KEY || apiKey !== env.CRASH_API_KEY) {
      return reply({ error: 'Unauthorized' }, 401);
    }

    // ── Parse & size-check body ───────────────────────────────────────────
    let body;
    try {
      const text = await request.text();
      if (text.length > MAX_BODY) return reply({ error: 'Payload too large' }, 413);
      body = JSON.parse(text);
    } catch {
      return reply({ error: 'Invalid JSON' }, 400);
    }

    // ── Validate required fields ──────────────────────────────────────────
    for (const f of ['version', 'platform', 'exit_code', 'log_tail', 'fingerprint']) {
      if (body[f] == null) return reply({ error: `Missing: ${f}` }, 400);
    }

    // ── Server-side rate limiting (requires KV — see wrangler.toml) ───────
    if (env.CRASH_CACHE) {
      const key  = `fp:${body.fingerprint}`;
      const last = await env.CRASH_CACHE.get(key);
      if (last && Date.now() - Number(last) < KV_COOLDOWN * 1000) {
        return reply({ status: 'rate_limited' }, 429);
      }
      await env.CRASH_CACHE.put(key, String(Date.now()), { expirationTtl: KV_COOLDOWN });
    }

    // ── File GitHub issue ─────────────────────────────────────────────────
    const ghRes = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/issues`, {
      method: 'POST',
      headers: {
        Authorization:  `token ${env.GITHUB_TOKEN}`,
        'Content-Type': 'application/json',
        'User-Agent':   'STT-CrashReporter/1.0',
        Accept:         'application/vnd.github.v3+json',
      },
      body: JSON.stringify({
        title:  buildTitle(body),
        body:   buildBody(body),
        labels: ['crash-report', 'bug'],
      }),
    });

    if (!ghRes.ok) {
      const err = await ghRes.text();
      console.error('GitHub API error:', ghRes.status, err);
      return reply({ error: 'Failed to create issue', code: ghRes.status }, 502);
    }

    const issue = await ghRes.json();
    return reply({ status: 'ok', issue_url: issue.html_url, issue_number: issue.number });
  },
};

// ── Formatting ──────────────────────────────────────────────────────────────

function buildTitle(b) {
  const os   = String(b.platform ?? '').split(' ')[0] || 'Unknown';
  const code = b.exit_code === 0 ? 'clean exit' : `exit ${b.exit_code}`;
  return `[Crash] v${esc(b.version)} ${os} — ${code} (#${b.fingerprint})`;
}

function buildBody(b) {
  const rows = [
    ['Version',             esc(b.version)],
    ['Platform',            esc(b.platform)],
    ['Python',              esc(b.python_version  ?? '—')],
    ['Timestamp (UTC)',     esc(b.timestamp       ?? '—')],
    ['Exit Code',           `\`${b.exit_code}\``],
    ['Consecutive Crashes', String(b.consecutive_crashes ?? '—')],
    ['GPU Enabled',         b.gpu_enabled ? 'Yes' : 'No'],
    ['Model',               esc(b.whisper_model   ?? '—')],
    ['Audio Backend',       esc(b.audio_backend   ?? '—')],
    ['Fingerprint',         `\`${b.fingerprint}\``],
  ].map(([k, v]) => `| **${k}** | ${v} |`).join('\n');

  // Prevent log content from breaking the code fence
  const log = String(b.log_tail ?? '').replace(/```/g, "'''");

  return `## Crash Report

> ⚠️ Filed automatically by the STT watchdog.
> Log content has been sanitized — IPs, paths, and credentials are redacted.

| Field | Value |
|---|---|
${rows}

## Log Tail (sanitized)

\`\`\`
${log}
\`\`\`

---
*Auto-filed by [STT Watchdog](https://github.com/${GITHUB_REPO})*
`;
}

function esc(s) {
  return String(s).replace(/\|/g, '\\|').replace(/`/g, "'");
}

function reply(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
