# SIH26188 - OCR text extraction engine
#
# Uses Tesseract OCR via pytesseract to extract text from document images.
# Requires the Tesseract binary to be installed on the system.
# If Tesseract is not available, operations return a clear error rather than crashing.

import os
import logging

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

_OCR_TARGET_MAX_DIMENSION = 800
_OCR_MAX_SCALE = 3.0


class OCREngine:
    """
    OCR engine for extracting text from document images.

    Uses Tesseract via pytesseract. Falls back gracefully if Tesseract
    is not installed, returning structured error information.

    Preprocessing pipeline:
        1. Load image via Pillow (supports png, jpg, bmp, tiff)
        2. Convert to grayscale via OpenCV (if available)
        3. Apply adaptive thresholding for better OCR accuracy
        4. Run Tesseract OCR
    """

    def __init__(self, tesseract_cmd=None):
        """
        Initialize the OCR engine.

        Args:
            tesseract_cmd: Optional path to the Tesseract binary.
                           If None, uses the system PATH.
        """
        if tesseract_cmd and _TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def is_available(self):
        """
        Check if the OCR engine is operational.

        Returns:
            Dictionary with 'available' bool and 'detail' string.
        """
        if not _PIL_AVAILABLE:
            return {"available": False, "detail": "Pillow (PIL) is not installed"}
        if not _TESSERACT_AVAILABLE:
            return {"available": False, "detail": "pytesseract is not installed"}

        # Check if the Tesseract binary is actually accessible
        try:
            version = pytesseract.get_tesseract_version()
            return {"available": True, "detail": f"Tesseract {version}"}
        except Exception:
            return {
                "available": False,
                "detail": "Tesseract binary not found. Install Tesseract OCR and add it to PATH.",
            }

    def extract_text(self, image_path):
        """
        Extract text from a document image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Dictionary with:
                - success (bool): Whether OCR completed successfully.
                - text (str|None): Extracted text, or None on failure.
                - error (str|None): Error message if OCR failed.
                - engine (str): OCR engine identifier.
                - confidence (float|None): Average OCR confidence (0-100),
                                           or None if not computable.
        """
        result = {
            "success": False,
            "text": None,
            "error": None,
            "engine": "tesseract",
            "confidence": None,
        }

        # --- Validate file exists ---
        if not os.path.isfile(image_path):
            result["error"] = "Image file not found"
            return result

        # --- Check Tesseract availability ---
        availability = self.is_available()
        if not availability["available"]:
            result["error"] = availability["detail"]
            return result

        try:
            # --- Load and preprocess a working copy ---
            image = Image.open(image_path)

            # Convert palette/RGBA images to RGB for consistency
            if image.mode in ('P', 'RGBA', 'LA'):
                image = image.convert('RGB')

            original_size = image.size
            image = self._prepare_working_image(image)
            was_upscaled = image.size != original_size

            # Grayscale preserves stroke gradients for Tesseract LSTM; use as primary
            preprocessed = image.convert('L')

            # --- Run Tesseract OCR ---
            raw_text = pytesseract.image_to_string(preprocessed)
            ocr_image = preprocessed
            if not raw_text.strip():
                # Try thresholded working-copy representation if grayscale returns no text.
                ocr_image = self._preprocess(image)
                raw_text = pytesseract.image_to_string(ocr_image)
            text = raw_text.strip()

            # --- Get confidence data ---
            confidence = self._get_confidence(ocr_image)

            result["success"] = True
            result["text"] = text if text else ""
            result["confidence"] = confidence

        except Exception as e:
            logger.exception("OCR extraction failed")
            result["error"] = f"OCR extraction failed: {str(e)}"

        return result

    @staticmethod
    def _prepare_working_image(pil_image):
        """Upscale small images conservatively without modifying the source file."""
        width, height = pil_image.size
        largest_dimension = max(width, height)
        if largest_dimension >= _OCR_TARGET_MAX_DIMENSION:
            return pil_image

        scale = min(
            _OCR_MAX_SCALE,
            _OCR_TARGET_MAX_DIMENSION / largest_dimension,
        )
        resized_size = (round(width * scale), round(height * scale))
        return pil_image.resize(resized_size, Image.Resampling.LANCZOS)

    def _preprocess(self, pil_image):
        """
        Preprocess a PIL image for better OCR accuracy.

        If OpenCV is available, applies grayscale conversion and
        adaptive thresholding. Otherwise, returns the original image.

        Args:
            pil_image: PIL Image object.

        Returns:
            Preprocessed PIL Image.
        """
        if not _CV2_AVAILABLE:
            # Fallback: just convert to grayscale via Pillow
            return pil_image.convert('L')

        # Convert PIL → OpenCV (numpy array)
        img_array = np.array(pil_image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Convert back to PIL
        return Image.fromarray(thresh)

    def _get_confidence(self, pil_image):
        """
        Compute the average OCR confidence score.

        Args:
            pil_image: Preprocessed PIL Image.

        Returns:
            Average confidence as a float (0-100), or None on failure.
        """
        try:
            data = pytesseract.image_to_data(
                pil_image, output_type=pytesseract.Output.DICT
            )
            confidences = [
                int(c) for c, text in zip(data['conf'], data['text'])
                if int(c) > 0 and text.strip()
            ]
            if confidences:
                return round(sum(confidences) / len(confidences), 1)
        except Exception:
            pass
        return None

