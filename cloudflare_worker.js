/**
 * STT Crash Reporter — Cloudflare Worker
 *
 * Receives sanitized crash reports from watchdog.py and files GitHub issues.
 *
 * ── Setup ──────────────────────────────────────────────────────────────────
 * 1. Deploy this worker (wrangler deploy, or paste into the Cloudflare dashboard)
 * 2. Add these secrets via: wrangler secret put <NAME>
 *    - GITHUB_TOKEN  : GitHub Personal Access Token
 *                      Fine-grained: "Issues: Read and write" on this repo
 *                      Classic:      "repo" scope (or just "public_repo")
 *    - CRASH_API_KEY : Any random string — must match crash_reporting.api_key in config.json
 * 3. (Optional) Create a KV namespace named CRASH_CACHE for server-side rate limiting
 *    wrangler kv:namespace create CRASH_CACHE
 *    then uncomment the [[kv_namespaces]] section in wrangler.toml
 * 4. Copy your Worker URL (https://stt-crash-reporter.<subdomain>.workers.dev)
 *    into config.json → crash_reporting.worker_url
 * ───────────────────────────────────────────────────────────────────────────
 */

const GITHUB_REPO   = 'zitlem/STT';
const MAX_BODY_SIZE = 65_536;         // 64 KB hard cap
const RATE_LIMIT_S  = 600;            // server-side cooldown per fingerprint (requires KV)

export default {
  async fetch(request, env) {
    if (request.method !== 'POST') {
      return reply({ error: 'Method not allowed' }, 405);
    }

    // ── Authentication ────────────────────────────────────────────────────
    const apiKey = request.headers.get('X-Api-Key') ?? '';
    if (!env.CRASH_API_KEY || apiKey !== env.CRASH_API_KEY) {
      return reply({ error: 'Unauthorized' }, 401);
    }

    // ── Parse body ────────────────────────────────────────────────────────
    let body;
    try {
      const text = await request.text();
      if (text.length > MAX_BODY_SIZE) {
        return reply({ error: 'Payload too large' }, 413);
      }
      body = JSON.parse(text);
    } catch {
      return reply({ error: 'Invalid JSON' }, 400);
    }

    // ── Validate required fields ──────────────────────────────────────────
    for (const field of ['version', 'platform', 'exit_code', 'log_tail', 'fingerprint']) {
      if (body[field] === undefined || body[field] === null) {
        return reply({ error: `Missing field: ${field}` }, 400);
      }
    }

    // ── Server-side rate limiting (requires KV binding: CRASH_CACHE) ──────
    if (env.CRASH_CACHE) {
      const key  = `fp:${body.fingerprint}`;
      const last = await env.CRASH_CACHE.get(key);
      if (last && Date.now() - Number(last) < RATE_LIMIT_S * 1000) {
        return reply({ status: 'rate_limited' }, 429);
      }
      await env.CRASH_CACHE.put(key, String(Date.now()), { expirationTtl: RATE_LIMIT_S });
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
      return reply({ error: 'Failed to create issue', detail: ghRes.status }, 502);
    }

    const issue = await ghRes.json();
    return reply({ status: 'ok', issue_url: issue.html_url, issue_number: issue.number });
  },
};

// ── Helpers ─────────────────────────────────────────────────────────────────

function buildTitle(b) {
  const os      = String(b.platform ?? '').split(' ')[0] || 'Unknown OS';
  const version = escText(b.version);
  const code    = b.exit_code === 0 ? 'clean exit' : `exit ${b.exit_code}`;
  return `[Crash] v${version} ${os} — ${code} (#${b.fingerprint})`;
}

function buildBody(b) {
  const rows = [
    ['Version',             escText(b.version)],
    ['Platform',            escText(b.platform)],
    ['Python',              escText(b.python_version ?? '—')],
    ['Timestamp (UTC)',     escText(b.timestamp ?? '—')],
    ['Exit Code',           `\`${b.exit_code}\``],
    ['Consecutive Crashes', String(b.consecutive_crashes ?? '—')],
    ['GPU Enabled',         b.gpu_enabled ? 'Yes' : 'No'],
    ['Model',               escText(b.whisper_model ?? '—')],
    ['Audio Backend',       escText(b.audio_backend ?? '—')],
    ['Fingerprint',         `\`${b.fingerprint}\``],
  ];

  const table = rows.map(([k, v]) => `| **${k}** | ${v} |`).join('\n');

  // Prevent the log content from breaking the code fence
  const logContent = String(b.log_tail ?? '').replace(/```/g, "'''");

  return `## Crash Report

> ⚠️ This issue was filed automatically by the STT watchdog.
> Log content has been sanitized (IPs, paths, and credentials redacted).

| Field | Value |
|---|---|
${table}

## Log Tail (last ~120 lines, sanitized)

\`\`\`
${logContent}
\`\`\`

---
*Auto-filed by [STT Watchdog](https://github.com/${GITHUB_REPO}) — please add reproduction steps if you can.*
`;
}

/** Escape characters that break Markdown table cells or inline code. */
function escText(s) {
  return String(s)
    .replace(/\|/g, '\\|')
    .replace(/`/g, "'");
}

function reply(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
