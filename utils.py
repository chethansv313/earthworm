import cv2
import numpy as np


def extract_leaf_with_grabcut(image_bytes):
    """
    Simplified GrabCut:
    Uses a very thin margin to avoid chopping off the edges of the leaf,
    and includes a safety check to return the original image if GrabCut
    accidentally deletes too much of it.

    Returns a tuple: (segmented_image, foreground_mask)
    foreground_mask is needed by is_valid_leaf() below to check whether
    the segmented object actually looks like a leaf.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None

    height, width = img.shape[:2]

    # 1. SIMPLIFIED MARGIN:
    # Use a 1-pixel border instead of 10% so we don't accidentally cut the leaf off.
    rect = (1, 1, width - 2, height - 2)

    mask = np.zeros((height, width), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        # 2. LESS AGGRESSIVE:
        # Reduce iterations from 5 to 3 for faster, softer background removal.
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 3. SAFETY NET:
        # If GrabCut erased more than 90% of the image, it means it failed to find
        # the foreground. Abort the cut and return the original image.
        kept_pixels = np.sum(mask2)
        total_pixels = height * width

        if (kept_pixels / total_pixels) < 0.10:
            print("DEBUG: GrabCut removed too much of the image. Defaulting to original.")
            # Even on fallback, return a mask (all 1s) so is_valid_leaf() has
            # something to check against.
            fallback_mask = np.ones((height, width), dtype=np.uint8)
            return img, fallback_mask

        segmented_img = img * mask2[:, :, np.newaxis]
        return segmented_img, mask2

    except Exception as e:
        print(f"GrabCut processing error: {e}")
        # Fallback to the original image if the OpenCV algorithm crashes
        fallback_mask = np.ones((height, width), dtype=np.uint8)
        return img, fallback_mask


def is_valid_leaf(image_bgr, foreground_mask, min_foreground_ratio=0.04, min_leaf_color_ratio=0.30):
    """
    Checks whether the GrabCut foreground actually looks like a leaf,
    BEFORE running the crop identifier or disease model on it.

    This is the missing safety check: your crop identifier is a 3-way
    classifier (rice/tomato/potato) with no "none of these" option, so
    it will confidently guess one of the three even on a photo of a
    car, a face, or a wall. This function catches that case first.

    Two checks:
      A) Foreground size - if GrabCut found almost nothing (blank wall,
         empty sky), there's no real subject at all.
      B) Color check - leaves are green/yellow/brown across their
         surface. A photo of an unrelated object will mostly fail
         this check.

    Returns: (is_valid: bool, reason: str)
    """
    total_pixels = foreground_mask.size
    foreground_pixels = int(foreground_mask.sum())
    foreground_ratio = foreground_pixels / total_pixels

    if foreground_ratio < min_foreground_ratio:
        return False, "No clear object detected in the image. Please upload a clearer photo of a leaf."

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Broad HSV range covering healthy green through yellowing/browning leaf tissue.
    lower_leaf = np.array([10, 35, 20])
    upper_leaf = np.array([95, 255, 255])
    leaf_color_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    combined_mask = cv2.bitwise_and(leaf_color_mask, leaf_color_mask, mask=foreground_mask)
    leaf_colored_pixels = int(np.count_nonzero(combined_mask))

    leaf_color_ratio = leaf_colored_pixels / (foreground_pixels + 1e-6)

    if leaf_color_ratio < min_leaf_color_ratio:
        return False, "The uploaded image doesn't appear to be a plant leaf. Please upload a clear leaf photo."

    return True, "OK"