# inference/motion.py
# Motion detection logic
# Computes motion ratio between current frame and background model

import cv2
import numpy as np


def compute_motion_ratio(
    gray_u8: np.ndarray,
    bg_u8: np.ndarray,
    diff_thresh: int = 15,
    dilate_kernel=None,
    dilate_iter: int = 1
) -> float:
    """
    Compute ratio of moving pixels between current frame and background.
    
    Args:
        gray_u8     : current grayscale frame (uint8)
        bg_u8       : background grayscale frame (uint8)
        diff_thresh : pixel difference threshold
        dilate_kernel: kernel for dilation (None to skip)
        dilate_iter : number of dilation iterations
    
    Returns:
        float: ratio of moving pixels (0.0 - 1.0)
    """
    diff = cv2.absdiff(gray_u8, bg_u8)
    _, fg = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    if dilate_kernel is not None and dilate_iter > 0:
        fg = cv2.dilate(fg, dilate_kernel, iterations=dilate_iter)

    nonzero = cv2.countNonZero(fg)
    return nonzero / float(gray_u8.size)


def update_background(
    bg_gray_f: np.ndarray,
    gray: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Update running background model using weighted average.

    Args:
        bg_gray_f : current background float32
        gray      : current grayscale frame uint8
        alpha     : update speed (0.0 - 1.0)

    Returns:
        np.ndarray: updated background float32
    """
    cv2.accumulateWeighted(gray.astype(np.float32), bg_gray_f, alpha)
    return bg_gray_f


def init_background(gray: np.ndarray) -> np.ndarray:
    """
    Initialize background model from first frame.

    Args:
        gray: first grayscale frame uint8

    Returns:
        np.ndarray: background float32
    """
    return gray.astype(np.float32)