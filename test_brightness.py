from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import timeit

from brightness.brightness import brightness
from brightness.contrast import contrast

# img = np.array(Image.open('test/lena.jpg'))
img = np.array(Image.open('test/big.jpg'))

print(img.shape)
plt.imshow(img)
plt.show()

start = timeit.default_timer()
# rgb = brightness(img, 50)
rgb = contrast(img, 1.5)
print('Time 1: {}'.format(timeit.default_timer() - start))

plt.imshow(rgb)
plt.show()
