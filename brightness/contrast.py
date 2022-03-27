from typing import Final

import numpy as np


def contrast(rgb: np.ndarray, f: float):
    input_shape = rgb.shape

    normalized: np.ndarray = (rgb / (np.ones(input_shape) * 255)).reshape(-1, 3)
    normalized = np.concatenate((normalized, np.ones((normalized.shape[0], 1), dtype=float)), axis=1)

    bias: Final[float] = 0.5 * (1.0 - f)
    bright_mat: Final[np.ndarray] = np.array([[f, 0, 0, 0],
                                              [0, f, 0, 0],
                                              [0, 0, f, 0],
                                              [bias, bias, bias, 1]])

    normalized = np.matmul(normalized, bright_mat)

    normalized[normalized > 1.0] = 1.0
    normalized[normalized < 0.0] = 0.0

    return normalized[:, :3].reshape(input_shape)
