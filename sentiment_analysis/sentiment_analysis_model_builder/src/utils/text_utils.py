"""Text utility functions for sentiment analysis."""

import re
import unicodedata
from typing import List, Optional

from loguru import logger


def clean_text(
    text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
) -> str:
    """Clean and normalize text content.

    Args:
        text: Input text to clean
        remove_html: Whether to remove HTML tags
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs
    if remove_urls:
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, "", text)

    # Remove email addresses
    if remove_emails:
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        text = re.sub(email_pattern, "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def normalize_text(
    text: str, lowercase: bool = True, normalize_unicode: bool = True
) -> str:
    """Normalize text content.

    Args:
        text: Input text to normalize
        lowercase: Whether to convert to lowercase
        normalize_unicode: Whether to normalize Unicode characters

    Returns:
        Normalized text
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize Unicode
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    return text


def validate_text_length(
    text: str, min_length: int = 10, max_length: int = 10000
) -> bool:
    """Validate text length constraints.

    Args:
        text: Text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        True if text meets length requirements
    """
    if not text:
        return False

    text_length = len(text.strip())
    return min_length <= text_length <= max_length


def extract_tickers(text: str) -> List[str]:
    """Extract ticker symbols from text.

    Args:
        text: Text to extract tickers from

    Returns:
        List of ticker symbols found
    """
    if not text:
        return []

    # Pattern for common ticker formats: $BTC, BTC, #BTC, etc.
    ticker_pattern = r"[\$#]?([A-Z]{2,5})"
    matches = re.findall(ticker_pattern, text.upper())

    # Remove duplicates while preserving order
    seen = set()
    tickers = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            tickers.append(match)

    return tickers


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.

    Args:
        text: Text to extract hashtags from

    Returns:
        List of hashtags found
    """
    if not text:
        return []

    hashtag_pattern = r"#(\w+)"
    hashtags = re.findall(hashtag_pattern, text)

    return hashtags


def extract_mentions(text: str) -> List[str]:
    """Extract mentions from text.

    Args:
        text: Text to extract mentions from

    Returns:
        List of mentions found
    """
    if not text:
        return []

    mention_pattern = r"@(\w+)"
    mentions = re.findall(mention_pattern, text)

    return mentions


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Number of words
    """
    if not text:
        return 0

    # Split by whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def count_characters(text: str, include_spaces: bool = True) -> int:
    """Count characters in text.

    Args:
        text: Text to count characters in
        include_spaces: Whether to include spaces in count

    Returns:
        Number of characters
    """
    if not text:
        return 0

    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))


def get_text_statistics(text: str) -> dict:
    """Get comprehensive text statistics.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            "characters": 0,
            "characters_no_spaces": 0,
            "words": 0,
            "lines": 0,
            "tickers": 0,
            "hashtags": 0,
            "mentions": 0,
        }

    cleaned_text = clean_text(text)

    stats = {
        "characters": count_characters(cleaned_text, include_spaces=True),
        "characters_no_spaces": count_characters(cleaned_text, include_spaces=False),
        "words": count_words(cleaned_text),
        "lines": len(cleaned_text.splitlines()),
        "tickers": len(extract_tickers(cleaned_text)),
        "hashtags": len(extract_hashtags(cleaned_text)),
        "mentions": len(extract_mentions(cleaned_text)),
    }

    return stats
