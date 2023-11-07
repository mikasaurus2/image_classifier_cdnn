# baseline model
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


def define_model():
    model = Sequential()

    # model.add() Adds a layer instance on top of the layer stack.
    model.add(
        # 2D convolution layer (e.g. spatial convolution over images).
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=(200, 200, 3),
        )
    )

    # Max pooling operation for 2D spatial data.
    # Downsamples the input along its spatial dimensions (height and width)
    # by taking the maximum value over an input window
    # (of size defined by `pool_size`) for each channel of the input.
    model.add(MaxPooling2D((2, 2)))

    # Flattens the input. Does not affect the batch size.
    model.add(Flatten())

    # Just your regular densely-connected NN layer.
    # `Dense` implements the operation:
    # `output = activation(dot(input, kernel) + bias)`
    # where `activation` is the element-wise activation function
    # passed as the `activation` argument, `kernel` is a weights matrix
    # created by the layer, and `bias` is a bias vector created by the layer
    # (only applicable if `use_bias` is `True`). These are all attributes of
    # `Dense`.
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    # sigmoid is a function that transforms any value in the domain to a number
    # between 0 and 1.
    # We want this here because we're doing a binary classification, so the answer
    # is some level between 0 and 1.
    model.add(Dense(1, activation="sigmoid"))

    # Gradient descent (with momentum) optimizer.
    opt = SGD(learning_rate=0.001, momentum=0.9)

    # Configures the model for training.
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")

    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")

    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()

def main():
    print("Preparing model...")
    model = define_model()

    # Create data generator. We need to scale the pixel values to the range
    # of 0-1. (ImageDataGenerator is deprecated; see docs. I'll continue to
    # use this for the tutorial.)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_iterator = datagen.flow_from_directory(
        "dataset_dogs_vs_cats/train/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200),
    )
    test_iterator = datagen.flow_from_directory(
        "dataset_dogs_vs_cats/test/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200),
    )

    # Fit the model. Once fit, the model can be evaluated on the test dataset directly.
    history = model.fit(
        train_iterator,
        steps_per_epoch=len(train_iterator),
        validation_data=test_iterator,
        validation_steps=len(test_iterator),
        epochs=20,
        verbose=0,
    )

    # Evaluate the model on a data generator.
    _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
    print("> %.3f" % (acc * 100.0))

    # plot learning curves
    summarize_diagnostics(history)

    return 0


if __name__ == "__main__":
    sys.exit(main())
