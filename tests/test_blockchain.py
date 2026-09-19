# SIH26188 - Blockchain and audit service tests

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blockchain.hashing import compute_hash
from blockchain.block import Block
from blockchain.ledger import Ledger
from blockchain.audit_service import AuditService


class TestHashing(unittest.TestCase):
    """Tests for the hashing utility."""

    def test_hash_deterministic(self):
        """Same input should always produce the same hash."""
        h1 = compute_hash("test data")
        h2 = compute_hash("test data")
        self.assertEqual(h1, h2)

    def test_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        h1 = compute_hash("input A")
        h2 = compute_hash("input B")
        self.assertNotEqual(h1, h2)

    def test_hash_is_hex_sha256(self):
        """Hash output should be a 64-character hex string (SHA-256)."""
        h = compute_hash("anything")
        self.assertEqual(len(h), 64)
        # Should be valid hexadecimal
        int(h, 16)


class TestBlock(unittest.TestCase):
    """Tests for the Block class."""

    def test_block_creation(self):
        """Block should store index, data, and previous hash."""
        block = Block(1, {"event": "TEST"}, "0000")
        self.assertEqual(block.index, 1)
        self.assertEqual(block.data, {"event": "TEST"})
        self.assertEqual(block.previous_hash, "0000")
        self.assertIsNotNone(block.hash)
        self.assertIsNotNone(block.timestamp)

    def test_block_hash_consistency(self):
        """Block hash should match recomputation."""
        block = Block(1, {"event": "TEST"}, "0000")
        self.assertEqual(block.hash, block.compute_block_hash())

    def test_block_to_dict(self):
        """to_dict should return all block fields."""
        block = Block(1, {"event": "TEST"}, "0000")
        d = block.to_dict()
        self.assertIn('index', d)
        self.assertIn('timestamp', d)
        self.assertIn('data', d)
        self.assertIn('previous_hash', d)
        self.assertIn('hash', d)
        self.assertEqual(d['index'], 1)


class TestLedger(unittest.TestCase):
    """Tests for the Ledger class."""

    def test_genesis_block(self):
        """New ledger should start with exactly one genesis block."""
        ledger = Ledger()
        self.assertEqual(len(ledger.chain), 1)
        self.assertEqual(ledger.chain[0].index, 0)

    def test_add_block(self):
        """Adding a block should increase chain length."""
        ledger = Ledger()
        ledger.add_block({"event": "test"})
        self.assertEqual(len(ledger.chain), 2)
        self.assertEqual(ledger.chain[1].index, 1)

    def test_chain_links(self):
        """Each block's previous_hash should match the prior block's hash."""
        ledger = Ledger()
        ledger.add_block({"event": "A"})
        ledger.add_block({"event": "B"})
        self.assertEqual(ledger.chain[1].previous_hash, ledger.chain[0].hash)
        self.assertEqual(ledger.chain[2].previous_hash, ledger.chain[1].hash)

    def test_chain_valid(self):
        """A properly constructed chain should pass validation."""
        ledger = Ledger()
        ledger.add_block({"event": "A"})
        ledger.add_block({"event": "B"})
        ledger.add_block({"event": "C"})
        self.assertTrue(ledger.is_chain_valid())

    def test_chain_tamper_detection(self):
        """Modifying a block's data should invalidate the chain."""
        ledger = Ledger()
        ledger.add_block({"event": "original"})
        ledger.add_block({"event": "final"})

        # Tamper with the middle block's data
        ledger.chain[1].data = {"event": "TAMPERED"}
        self.assertFalse(ledger.is_chain_valid())

    def test_genesis_tamper_detection(self):
        """Changing the genesis data or anchor should invalidate the chain."""
        ledger = Ledger()
        ledger.chain[0].data = {"event": "TAMPERED"}
        self.assertFalse(ledger.is_chain_valid())

        ledger = Ledger()
        ledger.chain[0].previous_hash = "changed-anchor"
        self.assertFalse(ledger.is_chain_valid())

    def test_stored_hash_tamper_detection(self):
        """Changing a stored hash should invalidate the chain."""
        ledger = Ledger()
        ledger.add_block({"event": "A"})
        ledger.chain[1].hash = "0" * 64
        self.assertFalse(ledger.is_chain_valid())

    def test_previous_hash_link_tamper_detection(self):
        """Breaking a previous-hash link should invalidate the chain."""
        ledger = Ledger()
        ledger.add_block({"event": "A"})
        ledger.add_block({"event": "B"})
        ledger.chain[2].previous_hash = "broken-link"
        self.assertFalse(ledger.is_chain_valid())

    def test_removed_interior_block_detection(self):
        """Removing an interior block should break indexes and links."""
        ledger = Ledger()
        ledger.add_block({"event": "A"})
        ledger.add_block({"event": "B"})
        ledger.add_block({"event": "C"})
        del ledger.chain[2]
        self.assertFalse(ledger.is_chain_valid())

    def test_reordered_block_detection(self):
        """Reordering blocks should invalidate the sequence and links."""
        ledger = Ledger()
        ledger.add_block({"event": "A"})
        ledger.add_block({"event": "B"})
        ledger.chain[1], ledger.chain[2] = ledger.chain[2], ledger.chain[1]
        self.assertFalse(ledger.is_chain_valid())

    def test_empty_chain_is_invalid(self):
        """A chain with no genesis block is not valid."""
        ledger = Ledger()
        ledger.chain.clear()
        self.assertFalse(ledger.is_chain_valid())

    def test_get_chain(self):
        """get_chain should return a list of dictionaries."""
        ledger = Ledger()
        ledger.add_block({"event": "test"})
        chain = ledger.get_chain()
        self.assertIsInstance(chain, list)
        self.assertEqual(len(chain), 2)
        self.assertIsInstance(chain[0], dict)


class TestAuditService(unittest.TestCase):
    """Tests for the AuditService wrapper."""

    def test_log_event_creates_block(self):
        """Logging an event should add a block to the ledger."""
        service = AuditService()
        initial_length = len(service.ledger.chain)
        result = service.log_event("DOCUMENT_UPLOAD", {
            "document_id": "test-123",
            "file_type": "png",
            "action": "upload",
        })
        self.assertEqual(len(service.ledger.chain), initial_length + 1)
        self.assertIn('block_index', result)
        self.assertIn('block_hash', result)
        self.assertEqual(result['event_type'], 'DOCUMENT_UPLOAD')

    def test_log_event_no_pii(self):
        """Audit block data should contain metadata, not raw PII."""
        service = AuditService()
        service.log_event("DOCUMENT_UPLOAD", {
            "document_id": "test-456",
            "file_type": "jpg",
            "action": "upload",
        })

        # Get the last block's data
        last_block = service.ledger.chain[-1]
        block_data = last_block.data

        # The audit data should have event_type and metadata
        self.assertEqual(block_data['event_type'], 'DOCUMENT_UPLOAD')
        self.assertIn('metadata', block_data)
        metadata = block_data['metadata']
        self.assertEqual(metadata['document_id'], 'test-456')

        # Should NOT contain fields like 'name', 'address', 'photo', 'face_image'
        data_str = str(block_data)
        for pii_field in ['name', 'address', 'photo', 'face_image', 'ssn', 'aadhaar']:
            self.assertNotIn(pii_field, data_str.lower(),
                             f"Block data should not contain PII field: {pii_field}")

    def test_log_event_empty_type_raises(self):
        """Logging with an empty event type should raise ValueError."""
        service = AuditService()
        with self.assertRaises(ValueError):
            service.log_event("", {"document_id": "test"})

    def test_get_audit_log(self):
        """get_audit_log should return the full chain as dicts."""
        service = AuditService()
        service.log_event("TEST_EVENT", {"key": "value"})
        log = service.get_audit_log()
        self.assertIsInstance(log, list)
        # Genesis + 1 event = 2 blocks
        self.assertEqual(len(log), 2)

    def test_verify_integrity(self):
        """Chain integrity should pass after normal operations."""
        service = AuditService()
        service.log_event("EVENT_A", {"id": "1"})
        service.log_event("EVENT_B", {"id": "2"})
        self.assertTrue(service.verify_integrity())

    def test_multiple_events(self):
        """Multiple events should all be recorded in order."""
        service = AuditService()
        for i in range(5):
            service.log_event("EVENT", {"seq": i})
        # Genesis + 5 events = 6 blocks
        self.assertEqual(len(service.ledger.chain), 6)
        self.assertTrue(service.verify_integrity())


if __name__ == '__main__':
    unittest.main()

