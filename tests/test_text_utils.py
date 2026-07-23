"""Pure text helpers from the live transcription pipeline (stt/text_utils.py)."""

from stt.text_utils import (
    DEFAULT_WHISPER_HALLUCINATIONS,
    apply_profanity_filter,
    count_sentence_units,
    distribute_whisper_translation,
    extract_context_translation,
    filter_hallucinated_text,
    get_hallucination_phrases,
    is_fuzzy_duplicate,
    is_whisper_hallucination,
    normalize_for_hallucination_check,
    remove_overlapping_prefix,
    scope_whisper_translation,
    split_into_sentences,
)


class TestSplitIntoSentences:
    def test_complete_and_remainder(self):
        sentences, remainder = split_into_sentences("Hello world. How are you? I am")
        assert sentences == ["Hello world.", "How are you?"]
        assert remainder == "I am"

    def test_empty(self):
        assert split_into_sentences("") == ([], "")

    def test_only_remainder(self):
        sentences, remainder = split_into_sentences("no punctuation here")
        assert sentences == []
        assert remainder == "no punctuation here"

    def test_multi_punctuation(self):
        sentences, remainder = split_into_sentences("Really?! Yes... ok")
        assert sentences == ["Really?!", "Yes..."]
        assert remainder == "ok"


class TestCountSentenceUnits:
    def test_complete_sentences(self):
        assert count_sentence_units("One. Two!") == 2

    def test_trailing_fragment_counts_as_one(self):
        assert count_sentence_units("One. Two! and a bit") == 3

    def test_empty(self):
        assert count_sentence_units("") == 0

    def test_fragment_only(self):
        assert count_sentence_units("just a fragment") == 1


class TestFuzzyDuplicate:
    def test_exact_duplicate(self):
        assert is_fuzzy_duplicate("Hello world.", ["Hello world."])

    def test_near_duplicate(self):
        assert is_fuzzy_duplicate("Hello world!", ["Hello world."])

    def test_different_sentence(self):
        assert not is_fuzzy_duplicate(
            "The weather is nice today.", ["Hello world."]
        )

    def test_empty_history(self):
        assert not is_fuzzy_duplicate("Hello world.", [])


class TestRemoveOverlappingPrefix:
    def test_strips_repeated_prefix(self):
        prev = "and then we went to the store"
        new = "went to the store and bought milk"
        out = remove_overlapping_prefix(new, prev)
        assert out.strip() == "and bought milk"

    def test_no_overlap_untouched(self):
        out = remove_overlapping_prefix("completely new text here", "previous sentence words")
        assert out == "completely new text here"

    def test_full_overlap_returns_empty(self):
        prev = "we walked along the river bank"
        new = "along the river bank"
        assert remove_overlapping_prefix(new, prev) == ""

    def test_short_texts_untouched(self):
        assert remove_overlapping_prefix("too short", "also short") == "too short"


class TestFilterHallucinatedText:
    def test_strips_cjk_runs(self):
        assert filter_hallucinated_text("hello 你好世界 world") == "hello world"

    def test_plain_text_untouched(self):
        assert filter_hallucinated_text("hello world") == "hello world"

    def test_empty_passthrough(self):
        assert filter_hallucinated_text("") == ""
        assert filter_hallucinated_text(None) is None

    def test_hangul_and_kana(self):
        assert filter_hallucinated_text("okay 안녕하세요 then こんにちは done") == "okay then done"


class TestNormalizeForHallucinationCheck:
    def test_lowercase_and_punctuation(self):
        assert normalize_for_hallucination_check("Thank You, For Watching!") == "thank you for watching"

    def test_apostrophes_removed(self):
        assert normalize_for_hallucination_check("Don't stop") == "dont stop"
        assert normalize_for_hallucination_check("Don’t stop") == "dont stop"

    def test_whitespace_collapsed(self):
        assert normalize_for_hallucination_check("  a   b  ") == "a b"

    def test_empty(self):
        assert normalize_for_hallucination_check("") == ""
        assert normalize_for_hallucination_check(None) == ""


class TestGetHallucinationPhrases:
    def test_defaults_when_unconfigured(self):
        assert get_hallucination_phrases({}) == DEFAULT_WHISPER_HALLUCINATIONS

    def test_disabled_returns_empty(self):
        assert get_hallucination_phrases({"enabled": False, "phrases": ["x"]}) == []

    def test_custom_phrases_win(self):
        assert get_hallucination_phrases({"phrases": ["my phrase"]}) == ["my phrase"]


class TestIsWhisperHallucination:
    PHRASES = DEFAULT_WHISPER_HALLUCINATIONS

    def test_substring_match_case_insensitive(self):
        assert is_whisper_hallucination("Thanks for watching!", self.PHRASES)
        assert is_whisper_hallucination("THANK YOU FOR WATCHING.", self.PHRASES)

    def test_punctuation_insensitive(self):
        assert is_whisper_hallucination("Don’t forget to subscribe...", self.PHRASES)

    def test_normal_speech_passes(self):
        assert not is_whisper_hallucination("And now we turn to the reading.", self.PHRASES)

    def test_empty_inputs(self):
        assert not is_whisper_hallucination("", self.PHRASES)
        assert not is_whisper_hallucination("anything", [])


class TestApplyProfanityFilter:
    CFG = {"enabled": True, "words": ["darn", "heck"], "replacement": "****"}

    def test_replaces_whole_words_case_insensitive(self):
        assert apply_profanity_filter("What the Darn is that", self.CFG) == "What the **** is that"

    def test_word_boundary_respected(self):
        # "darning" contains "darn" but is not a whole-word match
        assert apply_profanity_filter("she was darning socks", self.CFG) == "she was darning socks"

    def test_disabled_passthrough(self):
        cfg = dict(self.CFG, enabled=False)
        assert apply_profanity_filter("what the darn", cfg) == "what the darn"

    def test_no_words_passthrough(self):
        assert apply_profanity_filter("what the darn", {"enabled": True, "words": []}) == "what the darn"

    def test_custom_replacement(self):
        cfg = dict(self.CFG, replacement="[bleep]")
        assert apply_profanity_filter("oh heck", cfg) == "oh [bleep]"

    def test_empty_text(self):
        assert apply_profanity_filter("", self.CFG) == ""


class TestExtractContextTranslation:
    def test_no_context_returns_all(self):
        assert extract_context_translation("Todo el texto.", 0) == "Todo el texto."

    def test_empty_translation_fails(self):
        assert extract_context_translation("", 2) is None

    def test_exact_sentence_alignment(self):
        combined = "Contexto uno. Contexto dos. La meta."
        assert extract_context_translation(combined, 2) == "La meta."

    def test_remainder_counts_as_unit(self):
        combined = "Contexto uno. la meta sin punto"
        assert extract_context_translation(combined, 1) == "la meta sin punto"

    def test_merged_sentences_proportional_fallback(self):
        # Translator merged 2 context sentences into 1 output sentence; the
        # char-ratio fallback should split at the boundary nearest the ratio.
        combined = "Contexto fusionado largo aqui. La meta corta."
        ratio = 30 / 45  # context is ~2/3 of the source chars
        assert extract_context_translation(combined, 2, ratio) == "La meta corta."

    def test_no_split_possible_returns_none(self):
        # One output unit, context expected, no ratio -> caller must retranslate
        assert extract_context_translation("Solo una frase.", 2) is None


class TestDistributeWhisperTranslation:
    def test_empty_rows(self):
        assert distribute_whisper_translation("text", []) == []

    def test_single_row_gets_everything(self):
        assert distribute_whisper_translation("all of it", ["src"]) == ["all of it"]

    def test_sentence_count_alignment(self):
        out = distribute_whisper_translation("First one. Second one.", ["a b", "c d"])
        assert out == ["First one.", "Second one."]

    def test_proportional_word_split(self):
        # 3 sentences vs 2 rows -> proportional split; every word kept, in order
        text = "uno dos tres cuatro. cinco seis! siete ocho"
        out = distribute_whisper_translation(text, ["one two three four five six", "seven eight"])
        assert len(out) == 2
        assert " ".join(out).split() == text.split()
        assert all(part for part in out)

    def test_too_few_words_kept_on_first_row(self):
        out = distribute_whisper_translation("solo", ["a", "b", "c"])
        assert out == ["solo", "", ""]


class TestScopeWhisperTranslation:
    TIMED = [(0.0, 2.0, "first part"), (2.0, 4.0, "second part"), (10.0, 14.0, "future tail")]

    def test_drops_segments_past_span(self):
        assert scope_whisper_translation(self.TIMED, 4.0) == "first part second part"

    def test_margin_includes_borderline(self):
        # midpoint 12.0 vs span_end 11.6 + default margin 0.5 -> excluded;
        # with a large margin it is included
        assert scope_whisper_translation(self.TIMED, 11.6, margin=1.0) == "first part second part future tail"

    def test_empty_or_no_span_returns_none(self):
        assert scope_whisper_translation([], 4.0) is None
        assert scope_whisper_translation(self.TIMED, None) is None

    def test_nothing_in_span_returns_none(self):
        assert scope_whisper_translation([(10.0, 14.0, "tail")], 2.0) is None


class TestContextPromptTruncation:
    """Mirrors the context-prompt tail truncation in the live loop:
    right-truncate, drop the leading partial word, discard if no space."""

    @staticmethod
    def truncate(prompt_tail, ctx_max_chars):
        if len(prompt_tail) > ctx_max_chars:
            prompt_tail = prompt_tail[-ctx_max_chars:]
            _cut = prompt_tail.find(" ")
            if _cut > 0:
                prompt_tail = prompt_tail[_cut + 1:]
            elif _cut < 0:
                prompt_tail = ""
        return prompt_tail

    def test_short_tail_untouched(self):
        assert self.truncate("short tail", 200) == "short tail"

    def test_partial_word_dropped(self):
        out = self.truncate("abcdefghij klm nop", 10)  # last 10 chars: "ij klm nop"
        assert out == "klm nop"

    def test_unbroken_run_discarded(self):
        assert self.truncate("a" * 300, 200) == ""
