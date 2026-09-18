# SIH26188 - Hashing utilities for blockchain audit chain

import hashlib


def compute_hash(data):
    """
    Compute SHA-256 hash of any data.

    Args:
        data: Any data that can be converted to a string.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(str(data).encode('utf-8')).hexdigest()

