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
