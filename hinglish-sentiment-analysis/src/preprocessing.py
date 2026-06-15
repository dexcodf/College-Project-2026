"""
Text preprocessing for Hinglish (Hindi-English code-mixed) social media text.

Design goals:
  * Be gentle - transformer tokenizers handle a lot of normalisation already,
    so we avoid aggressive stemming / stopword removal that would destroy
    code-mixed signal.
  * Strip social-media noise (URLs, @mentions, RTs, excessive whitespace).
  * Provide a single `clean_text` entry point reused by training, the API,
    and the Streamlit app so train/serve preprocessing never diverges.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, List

from .config import RAW_LABEL_ALIASES

# --------------------------------------------------------------------------- #
# Regex patterns (compiled once)
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")          # keep the word, drop the '#'
_RT_RE = re.compile(r"^\s*rt\b[:\s]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_PRINTABLE_RE = re.compile(r"[​-‏﻿]")  # zero-width chars
_MULTISPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")    # "sooooo" -> "soo"
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs / emoticons
    "\U00002600-\U000027BF"   # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"   # flags
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str, *, remove_emoji: bool = True, lower: bool = True) -> str:
    """
    Normalise a single Hinglish text string.

    Parameters
    ----------
    text : str
        Raw input text.
    remove_emoji : bool
        Drop emoji glyphs (sentiment-bearing but noisy for subword tokenizers).
    lower : bool
        Lowercase the text. IndicBERT / mBERT-uncased benefit from this.
    """
    if text is None:
        return ""
    text = str(text)

    text = html.unescape(text)            # &amp; -> &
    text = _HTML_TAG_RE.sub(" ", text)
    text = _RT_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)   # #happy -> happy
    text = _NON_PRINTABLE_RE.sub("", text)

    if remove_emoji:
        text = _EMOJI_RE.sub(" ", text)

    text = _REPEAT_CHAR_RE.sub(r"\1\1", text)   # collapse char floods
    text = _MULTISPACE_RE.sub(" ", text).strip()

    if lower:
        text = text.lower()
    return text


def clean_batch(texts: Iterable[str], **kwargs) -> List[str]:
    """Vectorised convenience wrapper over :func:`clean_text`."""
    return [clean_text(t, **kwargs) for t in texts]


def normalize_label(raw_label) -> str | None:
    """
    Map a raw dataset label to one of the canonical labels
    ('Negative' / 'Neutral' / 'Positive'). Returns None if unmappable.
    """
    if raw_label is None:
        return None
    key = str(raw_label).strip().lower()
    return RAW_LABEL_ALIASES.get(key)
