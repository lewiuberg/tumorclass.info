from pathlib import Path
from typing import Union

import numpy as np
from pyconfs import Configuration
from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.models import Functional, Sequential


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_PATH = f"{PROJECT_ROOT}/models"

cfg = Configuration.from_file(f"{PROJECT_ROOT.parent}/pyproject.toml")
CLASSES = cfg.env.classes
WIDTH_HEIGHT = (cfg.env.width, cfg.env.height)


def predict(
    model: Union[Sequential, Functional],
    model_name: str,
    path: str = MODELS_PATH,
    target_size: tuple = WIDTH_HEIGHT,
    classes: list = CLASSES,
) -> dict:
    """
    Predict the class of an image.

    Parameters
    ----------
    model : Union[Sequential, Functional]
        The model to use for prediction.
    model_name : str
        The name of the model.
    path : str, optional
        Path to the image, by default MODELS_PATH.
    target_size : tuple, optional
        The target size of the image, by default WIDTH_HEIGHT.
    classes : list, optional
        The list of classes, by default CLASSES.

    Returns
    -------
    dict
        The prediction of the model.
    """
    # load the image
    image = load_img(path=path, target_size=target_size)
    # convert the image to a numpy array
    image = img_to_array(image)
    # expand the shape of the array to include the batch dimension
    image = expand_dims(image, axis=0)
    # predict the class of the image
    prediction = model.predict(image)

    return {
        "model_name": model_name,
        "classification": classes[np.argmax(prediction[0])].upper(),
        "confidence": f"{np.max(prediction[0])*100:.2f}%",
        classes[0]: f"{prediction[0][0]*100:.2f}%",
        classes[1]: f"{prediction[0][1]*100:.2f}%",
        classes[2]: f"{prediction[0][2]*100:.2f}%",
    }


def predicts(
    models: dict,
    path: str = MODELS_PATH,
    target_size: tuple = WIDTH_HEIGHT,
    classes: list = CLASSES,
) -> dict:
    """
    Predict the class of an image.

    Parameters
    ----------
    models : dict
        The models to use for prediction.
    path : str, optional
        Path to the image, by default MODELS_PATH.
    target_size : tuple, optional
        The target size of the image, by default WIDTH_HEIGHT.
    classes : list, optional
        The list of classes, by default CLASSES.

    Returns
    -------
    dict
        The prediction of the model.
    """
    predictions = {}
    for model_name, model in models.items():
        temp_prediction = predict(
            model=model,
            model_name=model_name,
            path=path,
            target_size=target_size,
            classes=classes,
        )

        del temp_prediction["model_name"]

        predictions[model_name] = temp_prediction

    return predictions
