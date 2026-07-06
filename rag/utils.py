import re


def tokenize(text: str) -> list[str]:
    """
    Regex-based structured tokenizer: extracts alphanumeric codes/SKUs, lowercases,
    and constructs bigram phrase tokens for product names.
    """
    pattern = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
    tokens = pattern.findall(text.lower())

    # Add bigram phrase tokens for product names (shingles)
    phrases = []
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]}_{tokens[i + 1]}")
    return tokens + phrases
