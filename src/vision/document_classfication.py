# SIH26188 - Document type classification
#
# BASELINE IMPLEMENTATION: Keyword-based classification.
# No trained model or dataset is available at this stage.
# This classifier searches OCR text for document-type-specific keywords
# and returns the best match with a confidence score.
#
# LIMITATIONS:
#   - Accuracy depends entirely on OCR text quality.
#   - Cannot classify image-only documents (no text fallback yet).
#   - Confidence scores are heuristic, not learned probabilities.
#
# EXTENSIBILITY:
#   - The classify() interface is stable. Replace the internal logic
#     with a trained CNN/ViT model in a future phase without changing callers.
#   - Add new document types by extending DOCUMENT_KEYWORDS.

import logging
import re

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.15
AMBIGUITY_MARGIN = 0.05

# Keyword sets for each supported document type.
# Each entry maps a document type name to:
#   - keywords: terms whose presence in OCR text indicates this type
#   - weight: higher weight for more distinctive keywords
DOCUMENT_KEYWORDS = {
    "passport": {
        "keywords": [
            "passport", "passeport", "republic", "nationality",
            "date of birth", "date of issue", "date of expiry",
            "place of birth", "surname", "given name", "type", "code",
            "machine readable", "mrz", "passport number", "passport no",
        ],
        "strong": ["passport", "passeport"],
    },
    "aadhaar": {
        "keywords": [
            "aadhaar", "aadhar", "aadhaar card", "aadhaar number",
            "unique identification", "unique identification authority",
            "unique identification authority of india", "government of india",
            "enrolment", "enrolment number", "enrolment no", "enrollment number",
            "vid", "male", "female", "year of birth", "dob",
        ],
        "strong": [
            "aadhaar", "aadhar", "aadhaar card",
            "unique identification authority",
        ],
    },
    "pan_card": {
        "keywords": [
            "permanent account number", "income tax department",
            "pan", "govt of india", "government of india",
            "father", "mother",
        ],
        "strong": ["permanent account number", "income tax"],
    },
    "driving_license": {
        "keywords": [
            "driving licence", "driving license", "driver",
            "transport", "motor vehicle", "valid till",
            "class of vehicle", "cov", "non-transport",
            "licence no", "license no",
        ],
        "strong": ["driving licence", "driving license"],
    },
    "voter_id": {
        "keywords": [
            "election commission", "voter", "elector",
            "electoral", "epic", "photo identity card",
        ],
        "strong": ["election commission", "voter", "electoral"],
    },
}


class DocumentClassifier:
    """
    Keyword-based document type classifier.

    BASELINE APPROACH: Searches for keyword matches in OCR text.
    This is NOT a trained ML model. The interface is designed to be
    replaced by a CNN/ViT classifier in a future phase.

    Supported types: passport, aadhaar, pan_card, driving_license, voter_id.
    Returns 'unknown' if no type matches with sufficient confidence.
    """

    def __init__(self, keyword_config=None):
        """
        Initialize the classifier.

        Args:
            keyword_config: Optional custom keyword configuration.
                           Defaults to DOCUMENT_KEYWORDS.
        """
        self.keywords = keyword_config or DOCUMENT_KEYWORDS

    @staticmethod
    def _matches_keyword(text, keyword):
        """Match a whole keyword or phrase, allowing flexible whitespace."""
        escaped_keyword = re.escape(keyword.strip()).replace(r"\ ", r"\s+")
        return re.search(
            rf"(?<!\w){escaped_keyword}(?!\w)",
            text,
            flags=re.IGNORECASE,
        ) is not None

    @staticmethod
    def _is_ambiguous_short_keyword(keyword):
        """Exclude short standalone fragments common in unrelated text."""
        return ' ' not in keyword.strip() and len(keyword.strip()) < 5

    def classify(self, ocr_text, image_path=None):
        """
        Classify a document based on its OCR text.

        Args:
            ocr_text: Extracted text from the document.
            image_path: Optional path to the document image.
                       Reserved for future image-based classification.
                       Currently unused.

        Returns:
            Dictionary with:
                - document_type (str): Detected type or 'unknown'.
                - confidence (float): Confidence score (0.0 - 1.0).
                - method (str): Classification method used.
                - all_scores (dict): Scores for all evaluated types.
        """
        if not ocr_text or not ocr_text.strip():
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "method": "keyword_baseline",
                "all_scores": {},
            }

        text_lower = ocr_text
        scores = {}
        match_counts = {}

        for doc_type, config in self.keywords.items():
            usable_keywords = [
                keyword for keyword in config["keywords"]
                if not self._is_ambiguous_short_keyword(keyword)
            ]
            total_keywords = len(usable_keywords)

            # Count whole-word/whole-phrase matches, never raw substrings.
            matches = sum(
                1 for keyword in usable_keywords
                if self._matches_keyword(text_lower, keyword)
            )
            match_counts[doc_type] = matches
            score = 0.0
            if total_keywords > 0:
                score = matches / total_keywords

            # Boost for strong (highly distinctive) keywords
            strong_matches = sum(
                1 for keyword in config.get("strong", [])
                if self._matches_keyword(text_lower, keyword)
            )
            if strong_matches > 0:
                score = min(1.0, score + 0.3 * strong_matches)

            scores[doc_type] = round(score, 3)

        # Find the best match
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
        else:
            best_type = "unknown"
            best_score = 0.0

        # Require sufficient evidence and separation from the runner-up.
        ranked_scores = sorted(scores.values(), reverse=True)
        second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0
        is_ambiguous = (
            best_score > 0.0 and
            best_score - second_score <= AMBIGUITY_MARGIN
        )
        if (
            best_score < MIN_CONFIDENCE or
            match_counts.get(best_type, 0) == 0 or
            is_ambiguous
        ):
            best_type = "unknown"
            best_score = 0.0

        return {
            "document_type": best_type,
            "confidence": best_score,
            "method": "keyword_baseline",
            "all_scores": scores,
        }

