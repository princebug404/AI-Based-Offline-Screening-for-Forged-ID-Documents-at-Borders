# SIH26188 - Block structure for blockchain audit chain

import time
import json
from blockchain.hashing import compute_hash


class Block:
    """
    A single block in the audit chain.

    Each block contains an index, timestamp, audit data,
    the hash of the previous block, and its own hash.
    """

    def __init__(self, index, data, previous_hash):
        """
        Create a new block.

        Args:
            index: Position of this block in the chain.
            data: Audit event data (dict or string). Must NOT contain PII.
            previous_hash: Hash of the preceding block in the chain.
        """
        self.index = index
        self.timestamp = time.time()
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.compute_block_hash()

    def compute_block_hash(self):
        """
        Compute the SHA-256 hash of this block's contents.

        Returns:
            Hexadecimal hash string.
        """
        block_content = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}{self.previous_hash}"
        return compute_hash(block_content)

    def to_dict(self):
        """
        Serialize block to a dictionary.

        Returns:
            Dictionary representation of the block.
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }

