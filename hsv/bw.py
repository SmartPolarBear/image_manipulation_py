from typing import Final, Tuple

import numpy as np


def black_white(rgb: np.ndarray, w: Tuple[float, float, float] = (0.299, 0.587, 0.144)):
    input_shape = rgb.shape

    normalized: np.ndarray = (rgb / (np.ones(input_shape) * 255)).reshape(-1, 3)

    bw_mat: Final[np.ndarray] = np.array([[w[0], w[0], w[0]],
                                          [w[1], w[1], w[1]],
                                          [w[2], w[2], w[2]]])

    normalized = np.matmul(normalized, bw_mat)

    normalized[normalized > 1.0] = 1.0
    normalized[normalized < 0.0] = 0.0

    return normalized.reshape(input_shape)
