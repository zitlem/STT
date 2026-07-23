"""Translation glossary post-processing and the live translation cache.

Extracted from speech_to_text.py so they can be imported (and unit-tested)
without the monolith's import-time side effects. Stdlib-only. The glossary
dictionary is passed in already loaded; the thin wrapper in speech_to_text.py
handles config/session-override/file resolution.
"""

import re
import threading


def apply_glossary(text, source_lang, target_lang, dictionary):
    """Apply NLLB glossary post-processing replacements from a loaded dictionary.

    dictionary is the custom-dictionary object ({"glossary": {"en_to_es": {...}}}).
    """
    try:
        glossary_key = f"{source_lang}_to_{target_lang}"
        glossary = (dictionary or {}).get("glossary", {}).get(glossary_key, {})
        if not glossary:
            return text

        # Longest terms first so a short term can't clobber a longer one containing it.
        # Lookarounds instead of \b: terms may start/end with punctuation, where \b
        # silently fails. Lambda replacement so backslashes in the target aren't
        # treated as regex templates. Note: \w matches CJK, so terms embedded in
        # unspaced CJK runs won't match.
        for source_term, target_term in sorted(glossary.items(), key=lambda kv: -len(kv[0])):
            pattern = r"(?<!\w)" + re.escape(source_term) + r"(?!\w)"
            text = re.sub(pattern, lambda m, t=target_term: t, text, flags=re.IGNORECASE)

        return text
    except Exception as e:
        print(f"[WARNING] Glossary application failed: {e}")
        return text


class TranslationCache:
    """Cache translated segments to avoid re-translating"""

    def __init__(self, max_size=1000):
        self._cache = {}  # {segment_id: {original, translated, target_lang}}
        self._max_size = max_size
        self._lock = threading.Lock()
        self._target_lang = None

    def get(self, segment_id, original_text, target_lang, accept_stale_lang=False):
        """Get cached translation or None.
        If accept_stale_lang=True, return cached translation even if target_lang differs
        (used to avoid retranslating old segments after a hot language switch)."""
        with self._lock:
            entry = self._cache.get(segment_id)
            if entry and entry['original'] == original_text and entry['target_lang'] == target_lang:
                return entry['translated']
            if accept_stale_lang and entry and entry['original'] == original_text:
                return entry['translated']
            return None

    def _evict_if_full(self):
        """Remove oldest entries when full (insertion order). Caller must hold _lock."""
        if len(self._cache) >= self._max_size:
            oldest = list(self._cache.keys())[:100]
            for key in oldest:
                del self._cache[key]

    def set(self, segment_id, original_text, translated_text, target_lang):
        """Cache a translation"""
        with self._lock:
            self._evict_if_full()
            self._cache[segment_id] = {
                'original': original_text,
                'translated': translated_text,
                'target_lang': target_lang
            }

    def invalidate(self, segment_id):
        """Invalidate a specific cached translation (e.g., after correction)"""
        with self._lock:
            if segment_id in self._cache:
                del self._cache[segment_id]

    def set_with_extras(self, segment_id, original_text, translated_text, target_lang, confidence=None, alternatives=None):
        """Cache a translation with confidence and alternatives"""
        with self._lock:
            self._evict_if_full()
            self._cache[segment_id] = {
                'original': original_text,
                'translated': translated_text,
                'target_lang': target_lang,
                'confidence': confidence,
                'alternatives': alternatives or [],
            }

    def get_extras(self, segment_id):
        """Get cached confidence and alternatives for a segment"""
        with self._lock:
            entry = self._cache.get(segment_id)
            if entry:
                return {
                    'confidence': entry.get('confidence'),
                    'alternatives': entry.get('alternatives', []),
                }
            return None

    def clear(self):
        """Clear all cached translations"""
        with self._lock:
            self._cache.clear()
            print("[LIVE-TRANSLATION] Translation cache cleared")

    def get_size(self):
        """Get current cache size"""
        with self._lock:
            return len(self._cache)

    def max_segment_id(self):
        """Highest integer segment id currently cached, or 0 if none"""
        with self._lock:
            return max((sid for sid in self._cache if isinstance(sid, int)), default=0)
