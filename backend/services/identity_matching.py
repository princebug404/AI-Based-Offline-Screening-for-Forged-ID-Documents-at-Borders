"""Matching against the local synthetic identity-record dataset."""

import json
import re
from datetime import datetime


class SyntheticIdentityMatcher:
    """Compare extracted fields with explicitly synthetic sample records."""

    MATCH_FIELDS = ('document_type', 'document_number', 'name', 'date_of_birth')

    def __init__(self, records_path):
        self.records_path = records_path
        self.records = self._load_records(records_path)

    @staticmethod
    def _load_records(records_path):
        with open(records_path, encoding='utf-8') as records_file:
            dataset = json.load(records_file)

        if dataset.get('synthetic_only') is not True:
            raise ValueError('Identity record dataset must be marked synthetic_only')
        return dataset.get('records', [])

    def match(self, document_type, fields):
        """Return field outcomes without returning the compared values."""
        extracted = {
            'document_type': document_type,
            'document_number': fields.get('document_number'),
            'name': fields.get('name'),
            'date_of_birth': fields.get('date_of_birth'),
        }
        record = self._find_record(
            extracted['document_number'],
        )

        if record is None:
            return {
                'status': 'Unknown Record',
                'record_id': None,
                'synthetic_database': True,
                'fields': {field: 'Unavailable' for field in self.MATCH_FIELDS},
            }

        outcomes = {
            field: self._compare_field(field, extracted[field], record.get(field))
            for field in self.MATCH_FIELDS
        }
        status = 'Match' if all(value == 'Match' for value in outcomes.values()) else 'Mismatch'
        if 'Unavailable' in outcomes.values():
            status = 'Manual Review'

        return {
            'status': status,
            'record_id': record.get('record_id'),
            'synthetic_database': True,
            'fields': outcomes,
        }

    def _find_record(self, document_number):
        if not document_number:
            return None

        normalized_number = self._normalize_identifier(document_number)
        for record in self.records:
            if self._normalize_identifier(record.get('document_number')) == normalized_number:
                return record
        return None

    @classmethod
    def _compare_field(cls, field, extracted, expected):
        if extracted in (None, '') or expected in (None, ''):
            return 'Unavailable'
        if field == 'document_type':
            matches = cls._normalize_text(extracted) == cls._normalize_text(expected)
        elif field == 'document_number':
            matches = cls._normalize_identifier(extracted) == cls._normalize_identifier(expected)
        elif field == 'date_of_birth':
            matches = cls._normalize_date(extracted) == cls._normalize_date(expected)
        else:
            matches = cls._normalize_text(extracted) == cls._normalize_text(expected)
        return 'Match' if matches else 'Mismatch'

    @staticmethod
    def _normalize_text(value):
        return re.sub(r'\s+', ' ', str(value).strip()).casefold()

    @classmethod
    def _normalize_identifier(cls, value):
        return re.sub(r'[^a-z0-9]', '', cls._normalize_text(value))

    @staticmethod
    def _normalize_date(value):
        text = str(value).strip()
        for pattern in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d'):
            try:
                return datetime.strptime(text, pattern).strftime('%Y-%m-%d')
            except ValueError:
                pass

        match = re.fullmatch(r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})', text)
        if match:
            day, month, year = match.groups()
            return f'{year}-{int(month):02d}-{int(day):02d}'
        return SyntheticIdentityMatcher._normalize_text(text)
