"""Glossary post-processing and the live translation cache (stt/translation_utils.py)."""

from stt.translation_utils import TranslationCache, apply_glossary


def glossary(mapping, key="en_to_es"):
    return {"glossary": {key: mapping}}


class TestApplyGlossary:
    def test_no_dictionary_passthrough(self):
        assert apply_glossary("hello", "en", "es", None) == "hello"
        assert apply_glossary("hello", "en", "es", {}) == "hello"

    def test_wrong_language_pair_passthrough(self):
        d = glossary({"church": "iglesia"}, key="en_to_fr")
        assert apply_glossary("the church", "en", "es", d) == "the church"

    def test_simple_replacement_case_insensitive(self):
        d = glossary({"church": "iglesia"})
        assert apply_glossary("The Church is open", "en", "es", d) == "The iglesia is open"

    def test_word_boundaries(self):
        d = glossary({"art": "arte"})
        assert apply_glossary("the start of art", "en", "es", d) == "the start of arte"

    def test_longest_term_wins(self):
        d = glossary({"holy": "santo", "holy spirit": "espíritu santo"})
        assert apply_glossary("the holy spirit", "en", "es", d) == "the espíritu santo"

    def test_punctuation_edged_terms(self):
        # \b would fail on terms starting/ending with punctuation; lookarounds must not
        d = glossary({"St. Paul": "San Pablo"})
        assert apply_glossary("read St. Paul today", "en", "es", d) == "read San Pablo today"

    def test_backslashes_in_target_are_literal(self):
        d = glossary({"path": r"C:\new\1"})
        assert apply_glossary("the path here", "en", "es", d) == r"the C:\new\1 here"

    def test_bad_dictionary_shape_fails_open(self):
        assert apply_glossary("hello", "en", "es", {"glossary": "not-a-dict"}) == "hello"


class TestTranslationCache:
    def test_get_miss(self):
        c = TranslationCache()
        assert c.get(1, "hello", "es") is None

    def test_set_and_get(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        assert c.get(1, "hello", "es") == "hola"

    def test_changed_original_misses(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        assert c.get(1, "hello there", "es") is None

    def test_changed_language_misses_unless_stale_accepted(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        assert c.get(1, "hello", "fr") is None
        # Hot language switch: old segments may keep their stale-language text
        assert c.get(1, "hello", "fr", accept_stale_lang=True) == "hola"

    def test_invalidate(self):
        c = TranslationCache()
        c.set(1, "hello", "hola", "es")
        c.invalidate(1)
        assert c.get(1, "hello", "es") is None
        c.invalidate(99)  # unknown id is a no-op

    def test_clear_and_size(self):
        c = TranslationCache()
        c.set(1, "a", "x", "es")
        c.set(2, "b", "y", "es")
        assert c.get_size() == 2
        c.clear()
        assert c.get_size() == 0

    def test_eviction_drops_oldest_hundred(self):
        c = TranslationCache(max_size=150)
        for i in range(150):
            c.set(i, f"t{i}", f"x{i}", "es")
        c.set(150, "t150", "x150", "es")  # triggers eviction of ids 0..99
        assert c.get(0, "t0", "es") is None
        assert c.get(99, "t99", "es") is None
        assert c.get(100, "t100", "es") == "x100"
        assert c.get(150, "t150", "es") == "x150"

    def test_extras_round_trip(self):
        c = TranslationCache()
        c.set_with_extras(5, "hello", "hola", "es", confidence=0.9, alternatives=["buenas"])
        assert c.get(5, "hello", "es") == "hola"
        assert c.get_extras(5) == {"confidence": 0.9, "alternatives": ["buenas"]}

    def test_extras_default_empty(self):
        c = TranslationCache()
        c.set(5, "hello", "hola", "es")
        assert c.get_extras(5) == {"confidence": None, "alternatives": []}
        assert c.get_extras(99) is None

    def test_max_segment_id_ignores_non_int_keys(self):
        c = TranslationCache()
        assert c.max_segment_id() == 0
        c.set(3, "a", "x", "es")
        c.set("live", "b", "y", "es")
        c.set(7, "c", "z", "es")
        assert c.max_segment_id() == 7
