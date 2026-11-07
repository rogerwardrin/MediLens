import cv2
import numpy as np
from PIL import Image


def to_bgr(img):
    """Accept a PIL image or a numpy array, return a BGR numpy array."""
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def resize_long_side(gray, target=1600):
    """Resize so the longest side equals target while maintaining aspect ratio."""
    h, w = gray.shape[:2]
    scale = target / max(h, w)
    if abs(scale - 1) < 0.05:
        return gray
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)


def denoise(gray):
    """Remove speckles and sensor noise using Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )


def binarize(gray):
    """Adaptive thresholding to handle uneven lighting and shadows."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def deskew(gray):
    """Straighten a slightly tilted image. Check results by eye; if an image
    rotates the wrong way, flip the sign of `angle`."""
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 15:  # ignore tiny or suspicious angles
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(img, target=1600, do_deskew=False):
    """Return a dict of every stage so you can compare them."""
    stages = {}
    gray = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2GRAY)
    stages["gray"] = gray
    gray = resize_long_side(gray, target)
    stages["resized"] = gray
    if do_deskew:
        gray = deskew(gray)
        stages["deskewed"] = gray
    clean = denoise(gray)
    stages["denoised"] = clean
    stages["binary"] = binarize(clean)
    return stages

