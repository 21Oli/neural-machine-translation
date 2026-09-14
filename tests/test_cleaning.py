"""Unit tests for nmt.data.clean and nmt.data.normalize."""

import pytest

from nmt.data.clean import (
    clean_corpus,
    normalize_whitespace,
    remove_html_tags,
    remove_urls,
)
from nmt.data.normalize import normalize_amharic, normalize_english, normalize_text


# ---------------------------------------------------------------------------
# clean.py
# ---------------------------------------------------------------------------


class TestRemoveHtmlTags:
    def test_removes_basic_tags(self):
        assert remove_html_tags("<b>bold</b>") == "bold"

    def test_removes_nested_tags(self):
        assert remove_html_tags("<div><p>text</p></div>") == "text"

    def test_no_tags(self):
        assert remove_html_tags("plain text") == "plain text"

    def test_empty_string(self):
        assert remove_html_tags("") == ""


class TestRemoveUrls:
    def test_removes_http_url(self):
        result = remove_urls("Visit http://example.com for more.")
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_https_url(self):
        result = remove_urls("See https://example.com/path?q=1")
        assert "https" not in result

    def test_removes_www_url(self):
        result = remove_urls("Check www.example.com out.")
        assert "www.example.com" not in result

    def test_no_url(self):
        assert remove_urls("no url here") == "no url here"


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("a  b   c") == "a b c"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_replaces_tabs_and_newlines(self):
        assert normalize_whitespace("a\tb\nc") == "a b c"


class TestCleanCorpus:
    def _make_corpus(self):
        src = ["hello world", "hi", "a", "duplicate line", "duplicate line", ""]
        tgt = ["bonjour monde", "salut", "a", "ligne dupliquée", "ligne dupliquée", ""]
        return src, tgt

    def test_removes_empty_pairs(self):
        src, tgt = self._make_corpus()
        s, t = clean_corpus(src, tgt, min_len=1, max_len=100, remove_duplicates=False)
        assert all(x.strip() for x in s)
        assert all(x.strip() for x in t)

    def test_removes_duplicates(self):
        src, tgt = self._make_corpus()
        s, t = clean_corpus(src, tgt, min_len=1, max_len=100, remove_duplicates=True)
        pairs = list(zip(s, t))
        assert len(pairs) == len(set(pairs))

    def test_filters_by_min_length(self):
        src, tgt = self._make_corpus()
        s, t = clean_corpus(src, tgt, min_len=2, max_len=100, remove_duplicates=False)
        for sentence in s + t:
            assert len(sentence.split()) >= 2

    def test_filters_by_max_length(self):
        src = ["word " * 200]
        tgt = ["mot " * 200]
        s, t = clean_corpus(src, tgt, min_len=1, max_len=100)
        assert len(s) == 0

    def test_lengths_stay_aligned(self):
        src, tgt = self._make_corpus()
        s, t = clean_corpus(src, tgt)
        assert len(s) == len(t)

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(AssertionError):
            clean_corpus(["a", "b"], ["only one"])


# ---------------------------------------------------------------------------
# normalize.py
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_unicode_normalization(self):
        # NFC: composed form
        text = "\u00e9"  # é (precomposed)
        result = normalize_text(text, unicode_form="NFC")
        assert result == "\u00e9"

    def test_lowercase(self):
        result = normalize_text("Hello World", lowercase=True)
        assert result == "hello world"

    def test_strips_whitespace(self):
        result = normalize_text("  hello  ")
        assert result == "hello"

    def test_english_curly_quotes(self):
        result = normalize_text("\u201chello\u201d", lang="en")
        assert '"' in result

    def test_amharic_normalization(self):
        # ሃ (U+1203) should map to ሀ (U+1200)
        text = "\u1203\u1204"
        result = normalize_text(text, lang="am")
        assert "\u1200" in result
