from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import timeit

from rgb2hsv.rgb2hsv import rgb_to_hsv, hsv_to_rgb
from hsv.hsv_manipulation import yuv_transform

# img = np.array(Image.open('test/lena.jpg'))
img = np.array(Image.open('test/big.jpg'))

print(img.shape)
plt.imshow(img)
plt.show()

start = timeit.default_timer()
hsv = rgb_to_hsv(img)
hsv[:, :, 1] = hsv[:, :, 1] * 1.5
rgb = hsv_to_rgb(hsv)
print('Time 1: {}'.format(timeit.default_timer() - start))

plt.imshow(rgb)
plt.show()

start = timeit.default_timer()
rgb = yuv_transform(img, 0, 1.5, 1)
print('Time 2: {}'.format(timeit.default_timer() - start))

plt.imshow(rgb)
plt.show()
