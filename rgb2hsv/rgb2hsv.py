import numpy as np


def rgb_to_hsv(rgb: np.ndarray):
    input_shape = rgb.shape
    normalized: np.ndarray = (rgb / (np.ones(rgb.shape) * 255)).reshape(-1, 3)
    nr, ng, nb = normalized[:, 0], normalized[:, 1], normalized[:, 2]

    cmax: np.ndarray = np.maximum(np.maximum(nr, ng), nb)
    cmin: np.ndarray = np.minimum(np.minimum(nr, ng), nb)

    deltac: np.ndarray = cmax - cmin

    none0cmax = cmax.copy()  # avoid dividing by 0
    none0cmax[none0cmax == 0] = -1.0
    s = deltac / none0cmax
    s[s < 0] = 0.0

    deltac[deltac == 0] = 1.0  # avoid dividing by 0

    h = 4.0 + (ng - nr) / deltac
    h[ng == cmax] = 2.0 + (nr[ng == cmax] - nb[ng == cmax]) / deltac[ng == cmax]
    h[nr == cmax] = (nb[nr == cmax] - ng[nr == cmax]) / deltac[nr == cmax]
    h[cmax == cmin] = 0.0

    h = (h / 6.0) % 1.0
    return np.dstack([h, s, cmax]).reshape(input_shape)


def hsv_to_rgb(hsv):
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    return rgb.reshape(input_shape)
