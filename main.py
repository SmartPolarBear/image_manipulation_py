from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from rgb2hsv.rgb2hsv import rgb_to_hsv, hsv_to_rgb

img = np.array(Image.open('test/lena.jpg'))
print(img.shape)
plt.imshow(img)
plt.show()

hsv = rgb_to_hsv(img)
hsv[:, :, 1] = hsv[:, :, 1] * 1.5

rgb = hsv_to_rgb(hsv)

plt.imshow(rgb)
plt.show()
