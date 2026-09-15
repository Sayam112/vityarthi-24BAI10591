import cv2
import numpy as np


def validate_frames(previous_frame, current_frame):
    if previous_frame is None or current_frame is None:
        raise ValueError("Both frames are required.")

    if previous_frame.shape != current_frame.shape:
        raise ValueError("Both frames must have the same dimensions.")


def frame_difference(previous_frame, current_frame, threshold=30):
    validate_frames(previous_frame, current_frame)

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    _, motion = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return motion


def optical_flow(previous_frame, current_frame):
    validate_frames(previous_frame, current_frame)

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros_like(previous_frame)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
