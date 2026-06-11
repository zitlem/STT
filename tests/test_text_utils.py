"""Pure text helpers from the live transcription pipeline."""

from conftest import extract_definitions

ns = extract_definitions(
    "speech_to_text.py",
    ["split_into_sentences", "is_fuzzy_duplicate", "remove_overlapping_prefix"],
)
split_into_sentences = ns["split_into_sentences"]
is_fuzzy_duplicate = ns["is_fuzzy_duplicate"]
remove_overlapping_prefix = ns["remove_overlapping_prefix"]


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
