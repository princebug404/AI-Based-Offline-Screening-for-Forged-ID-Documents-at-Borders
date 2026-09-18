# SIH26188 - Flask application configuration

import os
from dotenv import load_dotenv

# Load environment variables from .env file at project root
# find_dotenv searches upward, but we explicitly point to the project root .env
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_dotenv_path = os.path.join(_project_root, '.env')
load_dotenv(_dotenv_path)


class Config:
    """
    Flask configuration class.

    Sensitive values are read from environment variables.
    Paths are resolved relative to the project root (SIH26188/).
    """

    # --- Security ---
    # MUST be changed in production. Read from SECRET_KEY env var.
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # --- Database ---
    # Relative path from project root; resolved to absolute at runtime.
    _db_relative = os.environ.get('DATABASE_PATH', 'database/sih26188.db')
    DATABASE_PATH = os.path.join(_project_root, _db_relative)

    # --- File uploads ---
    _upload_relative = os.environ.get('UPLOAD_FOLDER', 'storage/uploads')
    UPLOAD_FOLDER = os.path.join(_project_root, _upload_relative)

    # Maximum upload size: 16 MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Allowed document image file extensions (lowercase, without dot)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

    # Cosine similarity threshold for the pretrained face embedding comparison.
    FACE_MATCH_THRESHOLD = float(os.environ.get('FACE_MATCH_THRESHOLD', '0.45'))

    # Separate local dataset containing fictional prototype identity records.
    _synthetic_identity_relative = os.environ.get(
        'SYNTHETIC_IDENTITY_DB_PATH',
        'data/synthetic/sample_identity_records.json',
    )
    SYNTHETIC_IDENTITY_DB_PATH = os.path.join(
        _project_root,
        _synthetic_identity_relative,
    )

