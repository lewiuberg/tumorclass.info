import os
import pickle
import random
from os.path import join as p_join

from sklearn.model_selection import train_test_split


def mkdir_if_not_exists(*paths):
    """
    Make folder if not exists.

    Args:
        paths (str): Paths to make.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def pickler(filename, operation, data=None):
    if operation == "save":
        pickle_out = open(f"{filename}.pickle", "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
    elif operation == "load":
        pickle_in = open(f"{filename}.pickle", "rb")
        data = pickle.load(pickle_in)
        return data


def move_files(
    input_path,
    output_path,
    split=None,
    num_files=None,
    randomize=False,
    seed=42,
):

    files = os.listdir(input_path)

    if randomize:
        random.seed(seed)
        random.shuffle(files)

    if isinstance(output_path, list) and split > 0:
        train_files, test_files = train_test_split(
            files, test_size=split, random_state=seed
        )
        output_path_1, output_path_2 = output_path
        for file in train_files:
            os.rename(
                p_join(input_path, file),
                p_join(output_path_1, file),
            )
        for file in test_files:
            os.rename(
                p_join(input_path, file),
                p_join(output_path_2, file),
            )
    elif not isinstance(output_path, list) and split == 0:
        for file in files:
            os.rename(
                p_join(input_path, file),
                p_join(output_path, file),
            )

    elif isinstance(output_path, str):
        if num_files is None:
            num_files = len(files)
        else:
            files = files[:num_files]
        for file in files:
            os.rename(p_join(input_path, file), p_join(output_path, file))
    else:
        print("Check arguments!")


def rename_path_files(*paths, file_type="jpg", counter=0):
    for path in paths:
        try:
            files = sorted(
                os.listdir(path), key=lambda x: int(x.split(".")[0])
            )
        except ValueError:
            files = sorted(os.listdir(path))

        for filename in files:
            if filename.endswith(file_type):
                counter += 1
                # print(filename, counter)
                os.rename(
                    p_join(path, filename),
                    p_join(path, f"{counter}.{file_type}"),
                )
