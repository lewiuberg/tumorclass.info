def replace_text_in_file(file_path, text_to_replace, replacement_text):
    with open(file_path, "r") as f:
        content = f.read()
    new_content = content.replace(text_to_replace, replacement_text)
    with open(file_path, "w") as f:
        f.write(new_content)


def remove_sort_from_pycm_classes():
    import os
    import site

    from pycm import ConfusionMatrix

    SITE_PACKAGES_PATH = site.getsitepackages()
    PYCM_PATH = os.path.join(SITE_PACKAGES_PATH[0], "pycm")
    PYCM_OBJ_FILE = os.path.join(PYCM_PATH, "pycm_obj.py")
    print(PYCM_OBJ_FILE)

    replace_text_in_file(
        PYCM_OBJ_FILE,
        "self.classes = sorted(list(mapping.values()))",
        "self.classes = list(mapping.values())",
    )
