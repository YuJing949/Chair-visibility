import numpy as np
from skimage import color

def analyze_visual_contrast(image, mask):
    """
    Compute brightness and color contrast between masked object and background.

    Parameters:
        image: RGB image (H, W, 3)
        mask: Boolean mask (H, W), True = object

    Returns:
        dict with brightness_contrast, delta_e, and text results
    """
    chair_pixels = image[mask]
    background_pixels = image[~mask]

    def brightness(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    chair_brightness = np.mean([brightness(p) for p in chair_pixels])
    background_brightness = np.mean([brightness(p) for p in background_pixels])
    brightness_contrast = abs(chair_brightness - background_brightness)

    lab_image = color.rgb2lab(image)
    chair_lab = lab_image[mask]
    background_lab = lab_image[~mask]
    delta_e = np.linalg.norm(np.mean(chair_lab, axis=0) - np.mean(background_lab, axis=0))

    brightness_result = "Good brightness contrast" if brightness_contrast > 40 else "Low brightness contrast"
    color_result = "Good color contrast" if delta_e > 20 else "Low color contrast"

    return {
        "brightness_contrast": brightness_contrast,
        "delta_e": delta_e,
        "brightness_result": brightness_result,
        "color_result": color_result
    }

def overlay_mask(image, mask, color=(255, 0, 0)):
    """
    Overlay mask on the image with the given RGB color.
    """
    result = image.copy()
    result[mask] = color
    return result
