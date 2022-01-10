import os
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pycm import ConfusionMatrix
from tensorflow import expand_dims
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from termcolor import colored
from tools.file import mkdir_if_not_exists, p_join


def augment_data(
    path, files, out_path, label, gen, file_type="jpg", aug_num=4
):
    # if out_path does not exist, create it
    mkdir_if_not_exists(out_path)

    image_paths = [p_join(path, file) for file in files]

    for i in range(len(image_paths)):
        file = files[i]
        # replace first char of string with label
        # file = label + file[1:]
        if file.endswith(f".{file_type}"):
            image = np.expand_dims(plt.imread(p_join(image_paths[i])), axis=0)
            plt.imsave(
                p_join(
                    out_path,
                    f"{file.replace(f'.{file_type}', '')}_{0}.{file_type}",
                ),
                image[0],
            )

            aug_iter = gen.flow(image)
            aug_images = [
                next(aug_iter)[0].astype(np.uint8) for _ in range(aug_num)
            ]

            # save the augmented images to out_path appending the name with the index
            for j, aug_image in enumerate(aug_images):
                plt.imsave(
                    p_join(
                        out_path,
                        f"{file.replace(f'.{file_type}', '')}_{j+1}.{file_type}",
                    ),
                    aug_image,
                )


def predict_gen(
    model: Model,
    gen: ImageDataGenerator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate predictions for a given generator.

    Args:
        model (Model): The model to use for prediction.
        gen (ImageDataGenerator): Generator to use for prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
            Predictions, y_pred, y_true, target_names.
    """
    # Make predictions
    predictions = model.predict(x=gen, steps=len(gen), verbose=1)
    # Get predicted labels
    # Alternative: y_pred = np.argmax(predictions, axis=1)
    y_pred = predictions.argmax(axis=-1)
    # Get true labels
    y_true = gen.classes
    # Get target names
    target_names = list(gen.class_indices.keys())

    return predictions, y_pred, y_true, target_names


def predictor(path, image_width, image_height, model, classes):
    img = load_img(path, target_size=(image_width, image_height))
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)

    # Print classified name and score
    print(
        f"Classification: {classes[np.argmax(predictions[0])].upper()} "
        f"with {np.max(predictions[0])*100:.2f}% confidence.\n"
    )
    print("Ratio")
    print(
        f"Normal: {predictions[0][0]*100:.2f}%\n"
        f"LGG: {predictions[0][1]*100:.2f}%\n"
        f"HGG: {predictions[0][2]*100:.2f}%"
    )


def count_params(model):
    non_trainable_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights]
    )
    trainable_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.trainable_weights]
    )
    return {
        "non_trainable_params": non_trainable_params,
        "trainable_params": trainable_params,
    }


def confusion_matrix(
    y_pred=None,
    y_true=None,
    digit=5,
    labels=None,
    save=False,
    load=False,
    path=None,
    file_name=None,
) -> ConfusionMatrix:
    if not load and y_pred is not None and y_true is not None:
        cm = ConfusionMatrix(y_true, y_pred, digit=digit)
        if labels:
            cm.relabel(mapping={i: labels[i] for i in range(len(labels))})
        if save and path is not None and file_name is not None:
            cm.save_obj(os.path.join(path, f"{file_name}_cm"))
        return cm
    if (
        load
        and labels
        and not save
        and path is not None
        and file_name is not None
    ):
        old_cm = ConfusionMatrix(
            file=open(os.path.join(path, f"{file_name}_cm.obj"), "r")
        )
        old_lbl = old_cm.classes
        old_actual_vector = np.array(old_cm.actual_vector)
        old_predict_vector = np.array(old_cm.predict_vector)
        old_digit = old_cm.digit
        cm = ConfusionMatrix(
            actual_vector=old_actual_vector,
            predict_vector=old_predict_vector,
            digit=old_digit,
        )
        cm.relabel(mapping=dict(zip(old_lbl, labels)))
        return cm


def split_array(array_pred, array_true, classes):
    n_classes = len(classes)
    array_pred_0, array_pred_1, array_pred_2 = np.split(array_pred, n_classes)
    array_true_0, array_true_1, array_true_2 = np.split(array_true, n_classes)
    # find longest string in classes
    max_len = max([len(x) for x in classes])
    print(
        f"{classes[0]}: {' ' * (max_len - len(classes[0]))}{array_pred_0}"
        f"\ntrue: {' ' * (max_len - 4)}{array_true_0}\n\n"
        f"{classes[1]}: {' ' * (max_len - len(classes[1]))}{array_pred_1}"
        f"\ntrue: {' ' * (max_len - 4)}{array_true_1}\n\n"
        f"{classes[2]}: {' ' * (max_len - len(classes[2]))}{array_pred_2}"
        f"\ntrue: {' ' * (max_len - 4)}{array_true_2}\n\n"
    )


def compare_predictions(array_pred, array_true, classes):
    n_classes = len(classes)
    preds = np.split(np.array(array_pred).astype(str), len(classes))
    trues = np.split(np.array(array_true).astype(str), len(classes))

    for i in range(n_classes):
        for j in range(len(preds[i])):
            if preds[i][j] != trues[i][j]:
                preds[i][j] = "X"

    idx = [x for x in range(len(preds[0]))]

    lists = [[idx, preds[x], trues[x]] for x in range(n_classes)]

    for i, lsts in enumerate(lists):
        print(f"[{classes[i]}]")

        for j, lst in enumerate(lsts):
            for i in lst:
                if j == 1:
                    if i == "X":
                        print(
                            f"{colored(str(i).rjust(len(str(max(idx)))), 'red')}",
                            end=" ",
                        )
                    else:
                        print(
                            f"{colored(str(i).rjust(len(str(max(idx)))), 'green')}",
                            end=" ",
                        )
                else:
                    print(str(i).rjust(len(str(max(idx)))), end=" ")
            print()
        print()


def enumerate_oxford(x: Union[list, str]) -> Union[None, str]:
    """
    Return as string with Oxford comma.

    Args:
        x (Union[list, str]): List or string to be formated.

    Returns:
        Union[None, str]: Formated string. None if x is None.
    """
    if isinstance(x, str):
        x = x.replace(" ", "").replace(",", "")
    else:
        x = [str(item) for item in x]

    if len(x) == 0:
        return None
    if 0 < len(x) < 3:
        return ", ".join(x)
    else:
        first_part = x[:-1]
        last_part = x[-1]
        return ", ".join(first_part) + ", and " + last_part


def learning_rate_scheduler(
    epoch: int = 0,
    epoch_steps: Optional[List[int]] = None,
    initial_lr: Optional[float] = None,
    lr_steps: Optional[List[float]] = None,
    use_factor: bool = False,
    verbose: int = 0,
) -> Callable:
    """
    Learning rate scheduler.

    Args:
        epoch (int, optional): Current epoch. Defaults to 0.
        epoch_steps (Optional[List[int]], optional): List of epochs to apply
        initial_lr (Optional[float], optional): Initial learning rate.
        lr_steps (Optional[List[float]], optional): List of lr or lr factors to apply
        use_factor (bool, optional): Use lr factor instead of lr list. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Callable: [description]
    """
    if initial_lr is None:
        initial_lr = 0.1
    elif initial_lr is not None and lr_steps is None:
        initial_lr = initial_lr
        use_factor = True

    if epoch_steps is None:
        epoch_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    if lr_steps is None:
        lr_steps = [
            1e-1,
            0.5e-1,
            1e-2,
            0.5e-2,
            1e-3,
            0.5e-3,
            1e-4,
            0.5e-4,
            1e-5,
        ]

    if len(epoch_steps) < len(lr_steps):
        lr_steps = lr_steps[: len(epoch_steps)]
    elif len(epoch_steps) > len(lr_steps):
        epoch_steps = epoch_steps[: len(lr_steps)]

    if use_factor is False or (
        epoch_steps is not None
        and lr_steps is not None
        and initial_lr is None
        and use_factor
    ):
        print(
            f"[{colored(f'INFO', 'green')}]: "
            f"'Using 'lr_steps' as list, not factors."
        )

    def compute_lr(lr_step: float) -> float:
        """
        Compute learning rate.

        Args:
            lr_step (float): Learning rate step.

        Returns:
            float: Learning rate.
        """
        return initial_lr * lr_step if use_factor else lr_step  # type: ignore

    def print_step(num: float, step: str) -> None:
        """
        Print step.

        Args:
            num (float): Step number.
            step (str): Step name. Initial or Next.
        """
        digits_after = len(f"{num:.10f}".split(".")[1].lstrip("0").rstrip("0"))
        digits_before = len(f"{num:.10f}".split(".")[0])
        zero_count = len(f"{num:.10f}".split(".")[1]) - len(
            f"{num:.10f}".split(".")[1].lstrip("0")
        )

        if digits_before > 0 and digits_after == 0:
            zero_count = 0

        print(
            f"{step} learning rate: "
            f"{colored(f'{num:.{zero_count + digits_after}f}', 'red')}"
        )

    def learning_rate_schedule(epoch: int = epoch) -> float:
        """
        Learning rate scheduler.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Learning rate.
        """
        if epoch == 0:
            if verbose:
                print_step(initial_lr, "Initial")  # type: ignore
            return initial_lr
        elif epoch < epoch_steps[0]:
            return initial_lr

        for epoch_tipping_point, lr_step in zip(
            reversed(epoch_steps), reversed(lr_steps)  # type: ignore
        ):
            if epoch >= epoch_tipping_point:
                lr = compute_lr(lr_step)
                if verbose and epoch == epoch_tipping_point:
                    print_step(lr, "Next")
                return lr

    setattr(learning_rate_schedule, "epoch_steps", epoch_steps)
    setattr(learning_rate_schedule, "initial_lr", initial_lr)
    setattr(learning_rate_schedule, "lr_steps", lr_steps)
    setattr(learning_rate_schedule, "use_factor", use_factor)
    setattr(learning_rate_schedule, "verbose", verbose)

    return learning_rate_schedule


# def learning_rate_scheduler(
#     epoch=0,
#     epoch_steps=[5, 10, 15, 20],
#     initial_lr=1e-3,
#     lr_steps=[1e-1, 1e-2, 1e-3, 0.5e-3],
#     use_factor=False,
#     verbose=0,
# ):
#     def learning_rate_schedule(epoch=epoch):

#         lr = initial_lr

#         if use_factor:
#             if epoch >= epoch_steps[3]:
#                 lr *= lr_steps[3]
#             elif epoch >= epoch_steps[2]:
#                 lr *= lr_steps[2]
#             elif epoch >= epoch_steps[1]:
#                 lr *= lr_steps[1]
#             elif epoch >= epoch_steps[0]:
#                 lr *= lr_steps[0]
#         else:
#             if epoch >= epoch_steps[3]:
#                 lr = lr_steps[3]
#             elif epoch >= epoch_steps[2]:
#                 lr = lr_steps[2]
#             elif epoch >= epoch_steps[1]:
#                 lr = lr_steps[1]
#             elif epoch >= epoch_steps[0]:
#                 lr = lr_steps[0]

#         zero_count = len(f"{lr:.10f}".split(".")[1]) - len(
#             f"{lr:.10f}".split(".")[1].lstrip("0")
#         )

#         if verbose:
#             if lr == initial_lr:
#                 print(
#                     f"Initial learning rate: "
#                     f"{colored(f'{lr:.{zero_count + 1}f}', 'red')}"
#                 )
#             else:
#                 print(
#                     f"Next learning rate: "
#                     f"{colored(f'{lr:.{zero_count + 1}f}', 'red')}"
#                 )
#         return lr

#     setattr(learning_rate_schedule, "epoch_steps", epoch_steps)
#     setattr(learning_rate_schedule, "initial_lr", initial_lr)
#     setattr(learning_rate_schedule, "lr_steps", lr_steps)

#     return learning_rate_schedule
