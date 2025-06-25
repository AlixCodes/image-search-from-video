import cv2
import numpy as np

def compute_color_histogram(image_path: str, bins=(8, 8, 8)) -> list:
    """
    Compute a 3D color histogram (RGB) for an image.

    :param image_path: Path to the image file
    :param bins: Number of bins for each channel
    :return: Flattened and normalized histogram vector
    """
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Image not found or unreadable: {image_path}")


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])


    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()
