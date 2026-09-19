# SIH26188 - Field extraction from OCR text
#
# Extracts structured fields (name, DOB, document number, expiry date)
# from raw OCR text using regex patterns.
#
# BASELINE IMPLEMENTATION: This uses pattern matching, not a trained NER model.
# Accuracy depends heavily on document layout and OCR quality.
# Fields that cannot be extracted are represented as None, never invented.
#
# Designed to be extensible: add document-specific extractors in the future.

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Date patterns ---
# Matches common date formats found on identity documents:
#   DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
#   MM/DD/YYYY, YYYY-MM-DD, YYYY/MM/DD
#   DD MMM YYYY, DD MMMM YYYY (e.g., 15 Jan 2025, 15 January 2025)
_DATE_PATTERN = re.compile(
    r'\b('
    r'\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}'     # DD/MM/YYYY variants
    r'|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}'       # YYYY-MM-DD variants
    r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'  # DD Mon YYYY
    r')\b',
    re.IGNORECASE,
)

# --- Document number patterns ---
# Matches alphanumeric sequences typical of ID/passport/license numbers.
# Requires at least 1 letter and 1 digit, 6-20 characters total.
_DOC_NUMBER_PATTERN = re.compile(
    r'\b([A-Z]{1,3}\d{4,17}|\d{1,4}[A-Z]{1,3}\d{3,15})\b',
    re.IGNORECASE,
)

_PASSPORT_NUMBER_PATTERN = re.compile(
    r'\bpassport\s*(?:no\.?|number|#)\s*[:\-]?\s*'
    r'(?:[A-Z0-9]+\s+){0,5}'
    r'([A-Z0-9]+(?:-[A-Z0-9]+)+|[A-Z]\d{4,17})\b',
    re.IGNORECASE,
)

_AADHAAR_NUMBER_PATTERN = re.compile(
    r'(?<!\d)(\d{4}(?:[ -]\d{4}){2}|\d{12})(?!\d)'
)

# --- Name label patterns ---
# Looks for text following common labels like "Name:", "Surname:", "Given Name:", etc.
_NAME_LABEL_PATTERN = re.compile(
    r'(?:(?:sur)?name|given\s*name|full\s*name|nom)\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.\-]{1,60})',
    re.IGNORECASE,
)

# --- DOB label patterns ---
_DOB_LABEL_PATTERN = re.compile(
    r'(?:date\s*of\s*birth(?:\s*/\s*d\.?o\.?b\.?)?|d\.?o\.?b\.?|born|birth\s*date|'
    r'year\s*of\s*birth|y\.?o\.?b\.?|naissance)'
    r'\s*[:/\-]?[^\d\n]*?'
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
    r'|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'
    r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
    r'|\d{4})',
    re.IGNORECASE,
)

_AADHAAR_STOPWORDS = {
    'government of india', 'unique identification authority of india',
    'enrolment no', 'enrollment no', 'republic of india', 'income tax department',
    'election commission', 'transport department', 'driving licence', 'driving license',
    'male', 'female', 'transgender', 'aadhaar', 'aadhar', 'your aadhaar no',
    'help', 'signature', 'address', 'valid', 'qr code', 'authority',
}

# --- Expiry label patterns ---
_EXPIRY_LABEL_PATTERN = re.compile(
    r'(?:expir[yation]*\s*date|date\s*of\s*expir[yation]*|valid\s*(?:until|thru|through|till)|exp\.?)'
    r'\s*[:\-]?\s*'
    r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
    r'|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'
    r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
    re.IGNORECASE,
)


class FieldExtractor:
    """
    Extract structured fields from raw OCR text.

    BASELINE APPROACH: Uses regex pattern matching against common
    identity document layouts. Does not assume all documents share
    the same layout. Fields that cannot be found are set to None.

    Future enhancement: Replace with document-type-specific extractors
    or a trained NER model for higher accuracy.
    """

    def extract_fields(self, ocr_text, document_type=None):
        """
        Extract structured fields from OCR text.

        Args:
            ocr_text: Raw text from OCR engine.
            document_type: Optional detected document type string.
                          Can be used for type-specific extraction in the future.

        Returns:
            Dictionary with:
                - name (str|None): Detected person name, or None.
                - date_of_birth (str|None): Detected DOB, or None.
                - document_number (str|None): Detected document/ID number, or None.
                - expiry_date (str|None): Detected expiry date, or None.
                - all_dates (list[str]): All date-like strings found.
                - extraction_method (str): Description of how extraction was done.
                - field_count (int): Number of fields successfully extracted (non-None).
        """
        if not ocr_text or not ocr_text.strip():
            return {
                "name": None,
                "date_of_birth": None,
                "document_number": None,
                "expiry_date": None,
                "all_dates": [],
                "extraction_method": "regex_baseline",
                "field_count": 0,
            }

        name = self._extract_name(ocr_text, document_type)
        dob = self._extract_dob(ocr_text)
        doc_number = self._extract_document_number(ocr_text, document_type)
        expiry = self._extract_expiry_date(ocr_text)
        all_dates = self._extract_all_dates(ocr_text)

        fields = {
            "name": name,
            "date_of_birth": dob,
            "document_number": doc_number,
            "expiry_date": expiry,
            "all_dates": all_dates,
            "extraction_method": "regex_baseline",
        }

        # Count how many primary fields were actually extracted
        primary_keys = ["name", "date_of_birth", "document_number", "expiry_date"]
        fields["field_count"] = sum(1 for k in primary_keys if fields[k] is not None)

        return fields

    def _extract_name(self, text, document_type=None):
        """Extract a person name from labeled text patterns or document context."""
        match = _NAME_LABEL_PATTERN.search(text)
        if match:
            name = match.group(1).strip()
            # Clean up: remove trailing noise, limit to reasonable length
            name = re.sub(r'\s{2,}', ' ', name)  # collapse whitespace
            # Take only up to a newline or obvious noise
            name = name.split('\n')[0].strip()
            if len(name) >= 2:
                return name

        # Aadhaar cards and letters frequently omit an explicit "Name:" label.
        is_aadhaar = (
            document_type == 'aadhaar' or
            re.search(r'\b(?:aadhaar|aadhar|unique\s*identification|uidai)\b', text, re.IGNORECASE) is not None
        )
        if is_aadhaar:
            # Case A: Aadhaar letter format ("To\n<Name>\n[S/O...]")
            to_match = re.search(
                r'(?:\bTo\b)\s*\n+([A-Za-z][A-Za-z\s\.\-]{2,50})(?=\s*\n+\s*(?:[S/CDW]/?O|C/o|s/o|d/o|w/o|PO:|Village|Flat|House|Near|Opp|[0-9]))',
                text,
                re.IGNORECASE,
            )
            if to_match:
                cand = to_match.group(1).strip()
                if len(cand) >= 2 and cand.lower() not in _AADHAAR_STOPWORDS:
                    return cand

            # Case B: Standard Aadhaar card layout (Name above DOB/Gender/Aadhaar number)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            anchor_idx = None
            for i, line in enumerate(lines):
                if re.search(r'\b(?:dob|date\s*of\s*birth|year\s*of\s*birth|yob)\b', line, re.I):
                    anchor_idx = i
                    break
                if re.search(r'\b(?:male|female)\b', line, re.I) and anchor_idx is None:
                    anchor_idx = i
                if re.search(r'\b(?:your\s*aadhaar\s*no|aadhaar\s*no)\b', line, re.I) and anchor_idx is None:
                    anchor_idx = i

            if anchor_idx is not None:
                for i in range(anchor_idx - 1, -1, -1):
                    cand = lines[i].strip()
                    if re.match(r'^[A-Za-z][A-Za-z\s\.\-]{1,49}$', cand):
                        cand_lower = cand.lower()
                        if not any(sw in cand_lower for sw in _AADHAAR_STOPWORDS):
                            words = cand.split()
                            if 1 <= len(words) <= 5 and all(len(w) >= 2 for w in words):
                                return cand

        return None

    def _extract_dob(self, text):
        """Extract date of birth using labeled patterns."""
        match = _DOB_LABEL_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_expiry_date(self, text):
        """Extract expiry date using labeled patterns."""
        match = _EXPIRY_LABEL_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_document_number(self, text, document_type=None):
        """Extract a labeled passport or context-gated Aadhaar number."""
        passport_match = _PASSPORT_NUMBER_PATTERN.search(text)
        if passport_match:
            return passport_match.group(1).strip()

        is_aadhaar_text = (
            document_type == 'aadhaar' or
            re.search(r'\b(?:aadhaar|aadhar)\b', text, re.IGNORECASE)
        )
        if is_aadhaar_text:
            aadhaar_match = _AADHAAR_NUMBER_PATTERN.search(text)
            if aadhaar_match:
                return aadhaar_match.group(1).strip()

        # Preserve the original generic alphanumeric fallback for other documents.
        match = _DOC_NUMBER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_all_dates(self, text):
        """Extract all date-like strings found in the text."""
        matches = _DATE_PATTERN.findall(text)
        return [m.strip() for m in matches]

