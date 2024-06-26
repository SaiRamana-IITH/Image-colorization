import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow import keras
generator = keras.models.load_model('generator_model.h5')
optimizer = keras.optimizers.Adam(kwargs={})

# Compile the generator with the optimizer
generator.compile(loss='binary_crossentropy', optimizer=optimizer)
# Load a grayscale test image
gray_image = Image.open('lion.jpg').convert('L')
img_size = 120

# Resize and normalize the grayscale image array
gray_img_array = (np.asarray(gray_image.resize((img_size, img_size)))) / 255.
gray_img_array = gray_img_array.reshape((1, img_size, img_size, 1))

# Use the generator model to colorize the image
colorized_img_array = generator(gray_img_array).numpy()

# Convert the colorized image array to image format and display it
plt.figure(figsize=(10,10))
gray_image_plot = plt.subplot(1, 2, 1)
gray_image_plot.set_title('Grayscale Input', fontsize=16)
plt.imshow(gray_img_array.reshape((img_size, img_size)), cmap='gray')

colorized_image_plot = plt.subplot(1, 2, 2)
colorized_image_plot.set_title('Colorized Output', fontsize=16)
colorized_image = Image.fromarray((colorized_img_array[0] * 255).astype('uint8'))
plt.imshow(colorized_image)

plt.show()