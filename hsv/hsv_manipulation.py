from typing import Final

import numpy as np


def yuv_transform(rgb: np.ndarray, hue_shift: float, saturation_scale: float, value_scale: float):
    rgb = rgb / (np.ones(rgb.shape) * 255)

    yiq_mat: Final[np.matrix] = np.matrix([[0.299, 0.587, 0.144],
                                           [0.596, -0.274, -0.321],
                                           [0.211, -0.523, 0.311]])
    rgb_mat: Final[np.matrix] = np.matrix([[1, 0.956, 0.621],
                                           [1, -0.272, -0.647],
                                           [1, -1.107, 1.705]])

    hue_angle: Final[float] = hue_shift * np.pi / 180.0
    hue_cos: Final[float] = np.cos(hue_angle)
    hue_sin: Final[float] = np.sin(hue_angle)

    hue_mat: Final[np.matrix] = np.matrix([[1, 0, 0],
                                           [0, hue_cos, -hue_sin],
                                           [0, hue_sin, hue_cos]])

    sta_mat: Final[np.matrix] = np.matrix([[1, 0, 0],
                                           [0, saturation_scale, 0],
                                           [0, 0, saturation_scale]])

    val_mat: Final[np.matrix] = value_scale * np.identity(3)

    trans_mat: Final[np.matrix] = rgb_mat * hue_mat * sta_mat * val_mat * yiq_mat

    input_shape: Final = rgb.shape

    # A x = x^T A^T
    result = np.matmul(rgb.reshape(-1, 3), np.asarray(np.transpose(trans_mat)))

    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0

    return result.reshape(input_shape)
