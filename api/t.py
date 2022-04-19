import os

import numpy as np
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


CLASSES = ["normal", "lgg", "hgg"]
MODELS_PATH = (
    "/Users/lewiuberg/Documents/Data Science/Projects/tumorclass.info/models/"
)
MODEL_NAME = "cnn_densenet121"
TEST_PATHS = [
    "/Users/lewiuberg/Desktop/Skjermbilder/lgg1.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/lgg2.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/lgg3.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/lgg4.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/hgg1.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/hgg2.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/hgg3.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/hgg4.png",
    "/Users/lewiuberg/Desktop/Skjermbilder/hgg5.png",
]


def predictors(path, image_width, image_height, model, classes):
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


def predictor(path, image_width, image_height, model, classes):
    img = load_img(path, target_size=(image_width, image_height))
    # print(type(img))
    img_array = img_to_array(img)
    # print(type(img_array))
    img_array = expand_dims(img_array, 0)  # Create batch axis
    # print(type(img_array))

    predictions = model.predict(img_array)

    return {
        "classification": classes[np.argmax(predictions[0])].upper(),
        "confidence": f"{np.max(predictions[0])*100:.2f}%",
        # # use the CLASSES
        # CLASSES[0]: f"{predictions[0][0]*100:.2f}%",
        # CLASSES[1]: f"{predictions[0][1]*100:.2f}%",
        # CLASSES[2]: f"{predictions[0][2]*100:.2f}%",
    }


model = load_model(os.path.join(MODELS_PATH, f"{MODEL_NAME}_model.h5"))
# history = pickler(
#     os.path.join(MODELS_PATH, f"{MODEL_NAME}_model_history"), "load"
# )

# predictors(
#     path=TEST_PATHS[0],
#     image_width=224,
#     image_height=224,
#     model=model,
#     classes=CLASSES,
# )

# a = predictor(
#     path=TEST_PATHS[0],
#     image_width=224,
#     image_height=224,
#     model=model,
#     classes=CLASSES,
# )
# print("\n\n")
# print(TEST_PATHS[0])
# print(a)

for path in TEST_PATHS:
    # get the last part of the path
    name = path.split("/")[-1]
    a = predictor(
        path=path,
        image_width=224,
        image_height=224,
        model=model,
        classes=CLASSES,
    )
    print(f"{name}: {a['classification']} with {a['confidence']} confidence.")
