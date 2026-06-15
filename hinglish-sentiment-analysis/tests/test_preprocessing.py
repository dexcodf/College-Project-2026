"""Unit tests for the text-cleaning / label-normalisation layer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text, normalize_label  # noqa: E402


def test_clean_text_removes_urls_and_mentions():
    out = clean_text("Hello @user check https://example.com bhai!!!")
    assert "http" not in out
    assert "@user" not in out
    assert "hello" in out


def test_clean_text_keeps_hashtag_word():
    assert "happy" in clean_text("#happy mood")


def test_clean_text_collapses_repeats():
    assert clean_text("sooooo good") == "soo good"


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_clean_text_unescapes_html():
    assert "&" in clean_text("Tom &amp; Jerry")


def test_normalize_label_aliases():
    assert normalize_label("positive") == "Positive"
    assert normalize_label("NEG") == "Negative"
    assert normalize_label("neutral") == "Neutral"
    assert normalize_label("garbage") is None
