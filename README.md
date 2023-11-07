# Dogs vs Cats
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

Data set from:
https://www.kaggle.com/c/dogs-vs-cats/data


1. setup virtual environment and dependencies
    
    python -m venv env
    source env/bin/activate

    pip install -r requirements.txt

2. install GUI for python

    sudo apt-get install python3-tk

3. install tensorflow

    https://www.tensorflow.org/install/pip

4. install NVidia drivers and CUDA toolkit

    https://developer.nvidia.com/cuda-downloads

May need nvidia-gds package as well.

    sudo apt-get install nvidia-gds

Needed a compatible versino of CuDNN (for deep neural networks).

    https://developer.nvidia.com/rdp/cudnn-download

5. tensorflow still couldn't find my cuda drivers

    pip3 install nvidia-tensorrt

Then point `$PATH` at the CUDA binary and `$LD_LIBRARY_PATH` at the
tensorrt and `libcudnn` libraries. Note that the paths might be different
depending on installation.

    export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.8/site-packages/tensorrt
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu


# make a basic script to output some photos

Use `matplotlib` to plot some images.

Specifically, use `pyplot` functions to create plots and show some images
of cats and dogs.

`imread` reads an image from a file into an array and `imshow` displays
the data as an image.

See `basic_image_plot.py`.


# photo size
The basic plot script above will show that the images are different sizes.

To more easily train the model, the images should be the same size. Smaller
inputs (images) results in a model that's quicker to train.

Resize the images to 200 x 200 pixels.

These can be resized and loaded into memory all at once, but would require
about 12 Gigs of RAM. To avoid stressing the computer, we can resize
the images as we use them with Keras.

# preprocess photos in standard directories
To train the model, we'll want the images in a standard set of directories.
Under our top level dataset directory, we'll have `train` and `test` subdirectories.
Each of those will have `cats` and `dogs` subdirectories.


    dataset_dogs_vs_cats/
    ├── test
    │   ├── cats
    │   └── dogs
    └── train
        ├── cats
        └── dogs

25% of the dog and cat images will be set aside for testing.

See `setup_data.py` for the code to do this.

# develop a baseline Convolutional Neuroal Network model


# Reading List

## Initial Tutorial
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

## VGG Model (Visual Geometry Group)
This is a standard Convolutional Neural Network (CNN) architecture with multiple layers.
https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/

## CNN Model (Convolutional neural network)
Popular in compute vision.

https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
https://viso.ai/deep-learning/deep-neural-network-three-popular-types/
