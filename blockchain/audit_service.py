# SIH26188 - Blockchain audit service

import time
from datetime import datetime, timezone

from blockchain.ledger import Ledger


class AuditService:
    """
    Audit logging service backed by the blockchain ledger.

    Wraps the Ledger to provide structured audit event logging.
    All events are recorded as blocks in the chain for tamper-evident history.

    IMPORTANT: Never store raw identity documents, facial images, or
    sensitive personal information (PII) in audit records.
    Only store event metadata: event type, document ID, timestamps, and actions.
    """

    def __init__(self):
        """Initialize the audit service with a new ledger."""
        self.ledger = Ledger()

    def log_event(self, event_type, metadata):
        """
        Log an audit event to the blockchain ledger.

        Args:
            event_type: Type of event (e.g., 'DOCUMENT_UPLOAD', 'STATUS_CHANGE').
            metadata: Dictionary of event metadata.
                      Must NOT contain PII, raw documents, or facial images.

        Returns:
            Dictionary with the block index, hash, and event summary.

        Raises:
            ValueError: If event_type is empty.
            Exception: If the ledger operation fails.
        """
        if not event_type:
            raise ValueError("event_type cannot be empty")

        # Build the audit record (no PII)
        audit_data = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }

        block = self.ledger.add_block(audit_data)

        return {
            "block_index": block.index,
            "block_hash": block.hash,
            "event_type": event_type,
        }

    def get_audit_log(self):
        """
        Retrieve the full audit log.

        Returns:
            List of block dictionaries representing the audit history.
        """
        return self.ledger.get_chain()

    def verify_integrity(self):
        """
        Verify the integrity of the audit chain.

        Returns:
            True if the chain has not been tampered with, False otherwise.
        """
        return self.ledger.is_chain_valid()

