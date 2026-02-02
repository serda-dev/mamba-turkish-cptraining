"""Text preprocessing and filtering."""

import re
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Apply basic cleaning to text.
    
    - Strip leading/trailing whitespace
    - Normalize multiple newlines to double newlines
    - Normalize multiple spaces to single space
    """
    # Strip
    text = text.strip()
    
    # Normalize newlines (keep paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize spaces (but not newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    return text


def preprocess_texts(
    texts: Iterator[str],
    min_length: int = 50,
    apply_cleaning: bool = True,
) -> Iterator[str]:
    """
    Preprocess and filter text iterator.
    
    Args:
        texts: Iterator of raw text strings
        min_length: Minimum character length after cleaning
        apply_cleaning: Whether to apply text cleaning
        
    Yields:
        Cleaned and filtered text strings
    """
    total = 0
    kept = 0
    too_short = 0
    
    for text in texts:
        total += 1
        
        if apply_cleaning:
            text = clean_text(text)
        
        if len(text) < min_length:
            too_short += 1
            continue
        
        kept += 1
        yield text
    
    logger.info(
        f"Preprocessing complete: kept {kept}/{total} texts "
        f"(filtered {too_short} short texts < {min_length} chars)"
    )
