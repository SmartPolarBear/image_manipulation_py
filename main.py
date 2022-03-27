from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from rgb2hsv.rgb2hsv import rgb_to_hsv, hsv_to_rgb
from hsv.hsv_manipulation import yuv_transform

img = np.array(Image.open('test/lena.jpg'))
print(img.shape)
plt.imshow(img)
plt.show()

hsv = rgb_to_hsv(img)
shape = hsv.shape
hsv = hsv.reshape(-1, 3)
hsv[:, 1] = hsv[:, 1] * 1.5
rgb = hsv_to_rgb(rgb_to_hsv(img))

plt.imshow(rgb)
plt.show()

rgb = yuv_transform(img, 0, 1.5, 1)

plt.imshow(rgb)
plt.show()
