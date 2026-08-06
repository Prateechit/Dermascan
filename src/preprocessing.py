"""
OpenCV-based image preprocessing for skin-lesion images.

The pipeline cleans up a raw smartphone/webcam photo before it reaches the
deep-learning model:

    raw image
      -> resize to a working size
      -> hair removal (morphological black-hat + inpainting)
      -> illumination / colour normalisation (Shades-of-Gray colour constancy)
      -> denoise
      -> resize to model input + scale to [0, 1]

Every step uses OpenCV, which is one of the mandatory requirements of the
assignment.
"""

import cv2
import numpy as np

# The trained CNN expects this input size (matches train/train.py).
MODEL_INPUT_SIZE = (224, 224)


def remove_hair(image):
    """Remove dark hair strands that occlude the lesion.

    Uses a black-hat morphological operation to detect thin dark structures,
    then inpaints those regions so the underlying skin is reconstructed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return inpainted


def shades_of_gray(image, power=6):
    """Shades-of-Gray colour constancy.

    Normalises the colour cast caused by different lighting/cameras so the
    model sees consistent colours regardless of the capture device.
    """
    img = image.astype(np.float32)
    flat = np.power(img, power)
    means = np.power(np.mean(flat, axis=(0, 1)), 1.0 / power)
    means = np.where(means == 0, 1.0, means)
    gray = np.mean(means)
    scale = gray / means
    balanced = np.clip(img * scale, 0, 255).astype(np.uint8)
    return balanced


def preprocess(image_bgr, for_model=True):
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    image_bgr : np.ndarray
        Raw image in BGR order (as loaded by cv2.imread).
    for_model : bool
        If True, returns a float32 array of shape (1, 224, 224, 3) scaled to
        [0, 1], ready to feed to the model. If False, returns a uint8 BGR image
        suitable for display / saving (the "cleaned" picture).

    Returns
    -------
    np.ndarray
    """
    # 1. Standardise working resolution.
    working = cv2.resize(image_bgr, (450, 450), interpolation=cv2.INTER_AREA)

    # 2. Remove hair occlusions.
    working = remove_hair(working)

    # 3. Colour-constancy normalisation.
    working = shades_of_gray(working)

    # 4. Light denoising while keeping edges.
    working = cv2.bilateralFilter(working, d=5, sigmaColor=50, sigmaSpace=50)

    if not for_model:
        return working

    # 5. Resize to model input and scale.
    resized = cv2.resize(working, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)


def save_processed_preview(image_bgr, out_path):
    """Save the cleaned image so the UI can show a 'processed' preview."""
    cleaned = preprocess(image_bgr, for_model=False)
    cv2.imwrite(out_path, cleaned)
    return out_path
