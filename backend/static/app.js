const form = document.querySelector('#screening-form');
const fileInput = document.querySelector('#document-file');
const selectedFile = document.querySelector('#selected-file');
const submitButton = document.querySelector('#submit-button');
const statusMessage = document.querySelector('#status');
const errorMessage = document.querySelector('#error');
const resultPanel = document.querySelector('#result-panel');
const resultStatus = document.querySelector('#result-status');
const resultFields = document.querySelector('#result-fields');
const resultJson = document.querySelector('#result-json');
const cameraPreview = document.querySelector('#camera-preview');
const capturedFace = document.querySelector('#captured-face');
const faceCanvas = document.querySelector('#face-canvas');
const cameraPlaceholder = document.querySelector('#camera-placeholder');
const startCameraButton = document.querySelector('#start-camera');
const captureFaceButton = document.querySelector('#capture-face');
const retakeFaceButton = document.querySelector('#retake-face');
const stopCameraButton = document.querySelector('#stop-camera');
const cameraStatus = document.querySelector('#camera-status');
const cameraError = document.querySelector('#camera-error');
const faceFileInput = document.querySelector('#face-file');
const faceSource = document.querySelector('#face-source');
const removeFaceButton = document.querySelector('#remove-face');

let cameraStream = null;
let capturedFaceDataUrl = null;

const displayFields = [
    ['document_id', 'Document ID'],
    ['document_type', 'Document type'],
    ['confidence', 'Classification confidence'],
    ['extracted_fields', 'Extracted fields'],
    ['ocr_engine', 'OCR engine'],
    ['ocr_confidence', 'OCR confidence'],
    ['processing_error', 'Processing error'],
    ['processed_at', 'Processed at'],
    ['audit', 'Audit metadata'],
    ['face_comparison', 'Face comparison'],
    ['identity_match', 'Synthetic record comparison'],
];

fileInput.addEventListener('change', () => {
    selectedFile.textContent = fileInput.files[0]?.name || 'No image selected';
});

faceFileInput.addEventListener('change', uploadFaceImage);

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearMessage();

    const file = fileInput.files[0];
    if (!file) {
        showError('Choose a document image before uploading.');
        return;
    }

    const formData = new FormData(form);
    setBusy(true, 'Uploading document...');
    try {
        const uploadData = await request('/api/documents/upload', {
            method: 'POST',
            body: formData,
        });

        setStatus('Processing document...');
        const resultData = await request(`/api/documents/${uploadData.document_id}/process`, {
            method: 'POST',
        });
        resultData.audit = uploadData.audit;
        if (capturedFaceDataUrl) {
            setStatus('Comparing faces...');
            resultData.face_comparison = await compareCapturedFace(uploadData.document_id);
        }
        showResult(resultData);
        setStatus('Processing complete.');
    } catch (error) {
        showError(error.message);
        setStatus('');
    } finally {
        setBusy(false);
    }
});

async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || data.error || `Request failed (${response.status})`);
    }
    return data;
}

async function compareCapturedFace(documentId) {
    const faceResponse = await fetch(capturedFaceDataUrl);
    const faceBlob = await faceResponse.blob();
    const faceData = new FormData();
    faceData.append('face', faceBlob, 'live-face.png');
    return request(`/api/documents/${documentId}/compare-face`, {
        method: 'POST',
        body: faceData,
    });
}

function showResult(result) {
    resultPanel.hidden = false;
    resultStatus.textContent = result.processing_status || 'unknown';
    resultStatus.className = `badge ${result.processing_status === 'completed' ? 'success' : 'warning'}`;
    resultFields.replaceChildren();

    displayFields.forEach(([key, label]) => {
        if (result[key] === null || result[key] === undefined || result[key] === '') {
            return;
        }
        const term = document.createElement('dt');
        term.textContent = label;
        const description = document.createElement('dd');
        description.textContent = formatValue(result[key]);
        resultFields.append(term, description);
    });
    resultJson.textContent = JSON.stringify(result, null, 2);
}

function formatValue(value) {
    return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
}

function setBusy(isBusy, message) {
    submitButton.disabled = isBusy;
    fileInput.disabled = isBusy;
    setStatus(message || '');
}

function setStatus(message) {
    statusMessage.textContent = message;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}

function clearMessage() {
    errorMessage.textContent = '';
    errorMessage.hidden = true;
}

startCameraButton.addEventListener('click', startCamera);
captureFaceButton.addEventListener('click', captureFace);
retakeFaceButton.addEventListener('click', retakeFace);
stopCameraButton.addEventListener('click', stopCamera);
removeFaceButton.addEventListener('click', removeFaceImage);
window.addEventListener('pagehide', stopCamera);
window.addEventListener('beforeunload', stopCamera);

async function startCamera() {
    clearCameraError();

    if (!navigator.mediaDevices?.getUserMedia) {
        showCameraError('Camera access is not supported by this browser.');
        return;
    }

    stopCamera(false);
    setCameraStatus('Requesting camera access...');
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: false,
        });
        cameraPreview.srcObject = cameraStream;
        cameraPreview.hidden = false;
        capturedFace.hidden = true;
        cameraPlaceholder.hidden = true;
        captureFaceButton.disabled = false;
        retakeFaceButton.disabled = true;
        stopCameraButton.disabled = false;
        startCameraButton.disabled = true;
        setCameraStatus('Camera ready. Position your face and capture the image.');
    } catch (error) {
        cameraStream = null;
        setCameraStatus('');
        showCameraError(cameraErrorMessage(error));
    }
}

function captureFace() {
    if (!cameraStream || cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        showCameraError('Start the camera and wait for the live preview before capturing.');
        return;
    }

    faceCanvas.width = cameraPreview.videoWidth;
    faceCanvas.height = cameraPreview.videoHeight;
    faceCanvas.getContext('2d').drawImage(cameraPreview, 0, 0);
    capturedFaceDataUrl = faceCanvas.toDataURL('image/png');
    capturedFace.src = capturedFaceDataUrl;
    faceSource.textContent = 'Face input: camera capture';
    capturedFace.hidden = false;
    cameraPreview.hidden = true;
    captureFaceButton.disabled = true;
    retakeFaceButton.disabled = false;
    removeFaceButton.disabled = false;
    setCameraStatus('Face captured locally. The image has not been uploaded.');
}

function retakeFace() {
    if (!cameraStream) {
        return;
    }

    capturedFace.removeAttribute('src');
    capturedFaceDataUrl = null;
    capturedFace.hidden = true;
    cameraPreview.hidden = false;
    captureFaceButton.disabled = false;
    retakeFaceButton.disabled = true;
    setCameraStatus('Camera ready. Position your face and capture the image.');
}

function uploadFaceImage() {
    clearCameraError();
    const file = faceFileInput.files[0];
    if (!file) {
        return;
    }

    stopCamera(false);
    const reader = new FileReader();
    reader.onload = () => {
        capturedFaceDataUrl = reader.result;
        capturedFace.src = capturedFaceDataUrl;
        capturedFace.hidden = false;
        cameraPlaceholder.hidden = true;
        faceSource.textContent = `Face input: uploaded demo image (${file.name})`;
        captureFaceButton.disabled = true;
        retakeFaceButton.disabled = true;
        removeFaceButton.disabled = false;
        setCameraStatus('Uploaded face image ready. It will use the same comparison checks.');
    };
    reader.onerror = () => showCameraError('Unable to read the selected face image.');
    reader.readAsDataURL(file);
}

function removeFaceImage() {
    capturedFaceDataUrl = null;
    capturedFace.removeAttribute('src');
    capturedFace.hidden = true;
    faceFileInput.value = '';
    faceSource.textContent = 'No face image selected';
    removeFaceButton.disabled = true;
    cameraPlaceholder.hidden = false;
    setCameraStatus('Face image removed.');
}

function stopCamera(showStatus = true) {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }

    cameraPreview.srcObject = null;
    cameraPreview.hidden = true;
    capturedFace.hidden = true;
    capturedFace.removeAttribute('src');
    capturedFaceDataUrl = null;
    faceSource.textContent = 'No face image selected';
    faceFileInput.value = '';
    cameraPlaceholder.hidden = false;
    captureFaceButton.disabled = true;
    retakeFaceButton.disabled = true;
    stopCameraButton.disabled = true;
    removeFaceButton.disabled = capturedFaceDataUrl === null;
    startCameraButton.disabled = false;
    if (showStatus) {
        setCameraStatus('Camera stopped.');
    }
}

function cameraErrorMessage(error) {
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        return 'Camera permission was denied. Allow camera access in your browser settings and try again.';
    }
    if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        return 'No camera was found on this device.';
    }
    if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        return 'The camera is unavailable or already being used by another application.';
    }
    if (error.name === 'SecurityError') {
        return 'Camera access was blocked by the browser security policy. Use a secure origin or localhost.';
    }
    return 'Unable to access the camera. Check the device and browser permissions, then try again.';
}

function setCameraStatus(message) {
    cameraStatus.textContent = message;
}

function showCameraError(message) {
    cameraError.textContent = message;
    cameraError.hidden = false;
}

function clearCameraError() {
    cameraError.textContent = '';
    cameraError.hidden = true;
}
