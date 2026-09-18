# SIH26188 - Blockchain ledger (in-memory audit chain)

from blockchain.block import Block


class Ledger:
    """
    In-memory blockchain ledger for audit events.

    Maintains a chain of blocks starting with a genesis block.
    Each new block references the hash of the previous block,
    providing tamper-evident audit logging.

    Note: This ledger is in-memory and resets on restart.
    Persistent storage will be added in a future phase.
    """

    def __init__(self):
        """Initialize the ledger with a genesis block."""
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = Block(0, {"event": "GENESIS", "message": "Audit chain initialized"}, "0")
        self.chain.append(genesis)

    def add_block(self, data):
        """
        Add a new block to the chain.

        Args:
            data: Audit event data (dict or string). Must NOT contain PII.

        Returns:
            The newly created Block.
        """
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), data, previous_block.hash)
        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self):
        """
        Verify the integrity of the entire chain.

        Checks that:
        1. Each block's stored hash matches its recomputed hash.
        2. Each block's previous_hash matches the preceding block's hash.

        Returns:
            True if the chain is valid, False otherwise.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Recompute the current block's hash and compare
            if current.hash != current.compute_block_hash():
                return False

            # Verify the chain link
            if current.previous_hash != previous.hash:
                return False

        return True

    def get_chain(self):
        """
        Get the full chain as a list of dictionaries.

        Returns:
            List of block dictionaries.
        """
        return [block.to_dict() for block in self.chain]

