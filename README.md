# SIH26188: AI-Based Fake Identity & Document Screening System

SIH26188 is a Flask-based prototype for screening identity-document images. It combines OCR, keyword-based document classification, structured field extraction, comparison with fictional synthetic records, face similarity comparison, and a local hash-linked audit ledger.

This is a demonstration system. It does not verify real-world identity, prove that a document is genuine, or connect to government verification services.

## Problem and Objectives

Identity screening often requires several checks across a document image: readable text, document type, extracted fields, a portrait, and an auditable processing history. This prototype demonstrates how those checks can be coordinated in one local browser workflow.

The objectives are to:

- Accept supported identity-document images through a browser.
- Extract readable text and selected fields with OCR and regex rules.
- Classify common document types using a keyword baseline.
- Compare extracted fields with clearly fictional synthetic records.
- Compare a document portrait with a camera-captured or uploaded face image.
- Record upload metadata in a local, in-memory hash-linked audit ledger.

## Implemented Features

### Document upload and processing

- Browser interface served at `/`.
- Supported document image formats: PNG, JPG, JPEG, BMP, and TIFF.
- Upload endpoint: `POST /api/documents/upload` with multipart field `file`.
- Processing endpoint: `POST /api/documents/<document_id>/process`.
- Status endpoint: `GET /api/documents/<document_id>/status`.
- Stored-result endpoint: `GET /api/documents/<document_id>/result`.

### OCR and field extraction

- Tesseract OCR through `pytesseract`.
- Pillow and OpenCV preprocessing.
- Grayscale OCR fallback when thresholding returns no text.
- Regex extraction for name, date of birth, document number, expiry date, and date-like values.
- Raw OCR text is not returned or persisted in processing results.

### Document classification

The current classifier is a keyword baseline. It supports signals for:

- Passport
- Aadhaar-like documents
- PAN cards
- Driving licences
- Voter IDs

Its confidence is a heuristic score, not a probability of correctness.

### Synthetic identity matching

Processing compares available fields against `data/synthetic/sample_identity_records.json`.

- Records are explicitly fictional demonstration data.
- The dataset is not an authoritative government database.
- Formatting differences such as case, whitespace, separators, and supported date formats are normalized for comparison.
- Results include overall status and field-level `Match`, `Mismatch`, or `Unavailable` outcomes.
- Only the minimum synthetic match summary is persisted with the document result.

### Face comparison

- Browser camera capture and demonstration face-image upload are supported.
- Face comparison endpoint: `POST /api/documents/<document_id>/compare-face` with multipart field `face`.
- InsightFace `buffalo_l` provides pretrained face detection and embeddings through ONNX Runtime.
- A single portrait is detected and cropped from the document before comparison.
- Comparison uses cosine similarity and a configurable threshold.
- Missing, multiple, blurry, small, or invalid faces produce an uncertain/manual-review result.
- Captured/uploaded face images are processed in memory and are not stored by the comparison route.

This is a similarity signal only. It does not prove that the person is physically present, verify identity, or establish document authenticity. No liveness detection or face sample-database matching is implemented.

### Audit logging

- Upload metadata is recorded in an in-memory hash-linked ledger.
- The ledger uses the project’s existing SHA-256 hashing and block structure.
- `AuditService.verify_integrity()` checks the genesis block, block indexes, stored hashes, and previous-hash links.
- The ledger resets when the process restarts.

This is not a decentralized blockchain and does not provide tamper-proof security against an attacker who can rewrite the complete in-memory chain.

### Not currently integrated

Risk-related source modules exist under `src/risk/`, but no risk-scoring route is registered in the active Flask application. Camera capture, OCR, classification, synthetic matching, and face comparison are the active web workflow.

## Technology Stack

- Python 3.13: application runtime used by the verified commands.
- Flask: application factory, web page serving, and JSON APIs.
- SQLite: local document metadata and processing-result storage.
- Tesseract and `pytesseract`: OCR.
- Pillow and OpenCV: image loading and preprocessing.
- InsightFace and ONNX Runtime: pretrained face detection and embeddings.
- NumPy: image and embedding calculations.
- `python-dotenv`: loads optional project-root `.env` configuration.
- `requests`: used by the retained desktop API client.
- PySide6: retained desktop prototype dependency; the current primary interface is the Flask web UI.

## Windows Setup

### Prerequisites

- Windows with Python 3.13 available through the `py` launcher.
- Tesseract OCR installed separately and available on `PATH`.
- Internet access on first face-comparison setup if InsightFace must download the `buffalo_l` model cache.

### Install

From the project root:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application loads `.env` from the project root when present. Supported configuration variables are:

- `SECRET_KEY`
- `DATABASE_PATH`
- `UPLOAD_FOLDER`
- `FACE_MATCH_THRESHOLD`
- `SYNTHETIC_IDENTITY_DB_PATH`
- `FLASK_DEBUG` (disabled unless explicitly set to `1`, `true`, `yes`, or `on`)

Do not commit secrets or private identity images. The repository’s ignore rules exclude local environment files, databases, uploads, logs, and downloaded model artifacts.

## Run the Web Prototype

Verified startup command:

```bat
py -3.13 -m backend.app
```

Open the local interface at:

```text
http://127.0.0.1:5055/
```

Debug mode is disabled by default. Explicitly setting `FLASK_DEBUG=true` enables it for local troubleshooting only.

## Basic Demonstration

1. Start the application with the command above.
2. Open `http://127.0.0.1:5055/`.
3. Select a fictional synthetic document image and choose **Upload and process**.
4. Review OCR confidence, document type, redacted extracted fields, synthetic-record comparison status, and audit metadata.
5. For face comparison, either start the camera and capture a face or choose **Upload Face Image for Demo**.
6. Submit the document with the face input available. The result shows face counts, cosine similarity, threshold, and the comparison status.

Use only fictional test data. A document can process successfully while its synthetic record status is `Unknown Record` or `Manual Review`.

## Run Tests

From the project root:

```bat
py -3.13 -m unittest discover -s tests -v
```

The tests cover API behavior, OCR and classification rules, synthetic matching, face comparison, database persistence, audit-chain integrity, and error handling. Tests use temporary databases and synthetic/mock inputs where appropriate.

## Current Limitations and Future Work

- OCR and classification are heuristic baselines and depend on image quality and visible text.
- Face comparison requires exactly one detectable, sufficiently clear face in both the document portrait crop and the live/demo image.
- Synthetic records are fictional local fixtures, not authoritative identity data.
- The audit ledger is in memory and resets on restart; deleting only its final block cannot be detected without an external checkpoint.
- No government API, external identity lookup, liveness detection, authenticity guarantee, or decentralized consensus exists.
- Risk modules and other model directories are not integrated into the active web workflow.
- Future work could add durable audit storage, stronger document/field models, calibrated evaluation, liveness checks, and controlled external verification integrations.
