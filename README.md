# Image-colorization

Created a deep learning python project in which black and white images can be converted into color images using Generative Adversial Networks. I used TensorFlow and Keras modules in this project

I used two networks Generator and Discriminator. The Job of Generator is to create coloured images from greyscale images which are closer to real images. The job of discriminator is to identify whether the image is generated or original and give probability between 0 and 1. The generator takes this feedback and creates better images so that it can fool the discriminator.
