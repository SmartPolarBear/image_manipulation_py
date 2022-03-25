from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from rgb2hsv.rgb2hsv import rgb_to_hsv, hsv_to_rgb

img = np.array(Image.open('test/lena.jpg'))
print(img.shape)
plt.imshow(img)
plt.show()

hsv = rgb_to_hsv(img)
shape = hsv.shape
hsv.reshape(-1, 3)
h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
s = s * 2.0
hsv = np.dstack([h, s, v]).reshape(shape)

rgb = hsv_to_rgb(hsv)
plt.imshow(rgb)
