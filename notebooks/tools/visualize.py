import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def print_prefix(prefix="PATH", *args, **kwargs):
    print(f"[{prefix}]:", *args, **kwargs)


def visualize_generator(
    img_file, gen, img_n=4, nrows=None, ncols=None, figsize=(10, 10)
):
    # Set figure size
    plt.figure(figsize=figsize)

    # Load the requested image
    img = load_img(img_file)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)

    # Generat augumentations from the generator
    it = gen.flow(samples, batch_size=1)
    images = []
    for _ in range(img_n):
        batch = it.next()
        image = batch[0].astype("uint8")
        images.append(image)

    images = np.array(images)

    # Create a grid of images from the generator with nrows and ncols
    index, height, width, channels = images.shape

    if nrows is None:
        nrows = int(np.ceil(np.sqrt(img_n)))
    if ncols is None:
        ncols = int(np.ceil(img_n / nrows))

    grid = (
        images.reshape(nrows, ncols, height, width, channels)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, channels)
    )

    plt.axis("off")
    plt.imshow(grid)


def plot_batch(
    images,
    labels=None,
    figsize=(14, 6),
    rows=1,
    interpolation=False,
    argmax=False,
    as_type=None,
):
    warnings.simplefilter(action="ignore", category=FutureWarning)

    if type(images[0]) is np.ndarray:
        if as_type:
            images = np.array(images).astype(as_type)  # np.uint8
        else:
            images = np.array(images)
        # if images.shape[-1] != 3:
        #     images = images.transpose((0, 2, 3, 1))
    fig = plt.figure(figsize=figsize)
    cols = (
        len(images) // rows
        if len(images) % 2 == 0
        else len(images) // rows + 1
    )
    for i in range(len(images)):
        sub_plot = fig.add_subplot(rows, cols, i + 1)
        sub_plot.axis("off")
        if labels is not None:
            if argmax:
                sub_plot.set_title(np.argmax(labels[i]), fontsize=10)
            else:
                sub_plot.set_title(labels[i], fontsize=10)
        plt.imshow(images[i], interpolation=None if interpolation else "none")

    warnings.simplefilter(action="default", category=FutureWarning)


def plot_epochs(
    history,
    figsize=(14, 8),
    plot_lr=0,
    lr_x_offset=0.0,
    lr_y_offset=0.0,
    scale=1000,
    filepath=None,
):
    epochs = len(history.history["accuracy"])
    lbl_org = [x for x in range(epochs)]
    lbl_new = [x + 1 for x in range(epochs)]

    # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy
    plt.figure(figsize=figsize)
    plt.plot(
        history.history["accuracy"],
        color="g",
        linestyle="dashed",
        marker="o",
        markerfacecolor="g",
        markersize=5,
    )
    plt.plot(
        history.history["val_accuracy"],
        color="b",
        linestyle="dashed",
        marker="o",
        markerfacecolor="b",
        markersize=5,
    )

    if plot_lr == 1 or plot_lr == 3:
        lr_history = [x * scale for x in history.history["lr"]]
        plt.plot(
            lr_history,
            color="r",
            linestyle="dashed",
            marker="o",
            markerfacecolor="r",
            markersize=5,
        )

        last_z = 1

        for x, y, z in zip(lbl_org, lr_history, history.history["lr"]):
            if last_z != z:
                plt.text(
                    x + lr_x_offset,
                    y + lr_y_offset,
                    f"{z:.3e}",
                    ha="left",
                    va="bottom",
                    color="r",
                )
            last_z = z

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train_acc", "val_acc"])  # , loc="upper left")
    plt.xlim(-0.25, (max(lbl_org) + 0.25))
    plt.xticks(lbl_org, lbl_new)
    plt.ylim(0, 1)
    plt.grid()
    if filepath:
        plt.savefig(f"{filepath}_model_accuracy.png")
    plt.show()
    plt.close()

    max_loss = max(
        max(history.history["loss"]), max(history.history["val_loss"])
    )
    # summarize history for loss
    plt.figure(figsize=figsize)
    plt.plot(
        history.history["loss"],
        color="g",
        linestyle="dashed",
        marker="o",
        markerfacecolor="g",
        markersize=5,
    )
    plt.plot(
        history.history["val_loss"],
        color="b",
        linestyle="dashed",
        marker="o",
        markerfacecolor="b",
        markersize=5,
    )

    if plot_lr == 2 or plot_lr == 3:
        lr_history = [x * scale for x in history.history["lr"]]
        plt.plot(
            lr_history,
            color="r",
            linestyle="dashed",
            marker="o",
            markerfacecolor="r",
            markersize=5,
        )

        last_z = 1

        for x, y, z in zip(lbl_org, lr_history, history.history["lr"]):
            if last_z != z:
                plt.text(
                    x + lr_x_offset,
                    y + lr_y_offset,
                    f"{z:.3e}",
                    ha="left",
                    va="bottom",
                    color="r",
                )
            last_z = z

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train_loss", "val_loss"])  # , loc="upper left")
    plt.xlim(-0.25, (max(lbl_org) + 0.25))
    plt.xticks(lbl_org, lbl_new)
    if max_loss <= 1:
        plt.ylim(0, 1)
    else:
        plt.ylim(0, (max_loss + (max_loss * 0.1)))
    plt.grid()
    if filepath:
        plt.savefig(f"{filepath}_model_loss.png")
    plt.show()
    plt.close()
