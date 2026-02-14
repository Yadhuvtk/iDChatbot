"""
Input text normalizer for consistent matching.
Normalizes user input before embedding: lowercase, strip whitespace, remove punctuation.
"""
import re
import string


def normalize(text: str) -> str:
    """
    Normalize input text for consistent embedding and matching.

    Steps:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase
        3. Remove punctuation (except hyphens within words)
        4. Collapse multiple spaces into one
    """
    if not text:
        return ""

    # Strip whitespace
    text = text.strip()

    # Lowercase
    text = text.lower()

    # Remove punctuation but keep hyphens within words and spaces
    # Replace punctuation with space (except hyphens between alphanumerics)
    result = []
    for i, ch in enumerate(text):
        if ch in string.punctuation:
            # Keep hyphens between alphanumeric characters
            if ch == "-" and i > 0 and i < len(text) - 1:
                if text[i - 1].isalnum() and text[i + 1].isalnum():
                    result.append(ch)
                    continue
            # Replace other punctuation with space
            result.append(" ")
        else:
            result.append(ch)
    text = "".join(result)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text