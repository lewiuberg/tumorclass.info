# Tumorclass.info <!-- omit in toc -->

This repository contains the data set, preprocessing pipeline, CNN architecture implementations, model training and evaluation pipelines, and a simple web application for visualizing the results.

Please visit [Tumorclass.info](https://app.tumorclass.info) website to see it in action.

## Table of Contents <!-- omit in toc -->

- [Citation](#citation)
  - [APA](#apa)
  - [BibTex](#bibtex)
- [Preprocessing](#preprocessing)
- [Augmentation](#augmentation)
- [Model Training](#model-training)
- [User Interface](#user-interface)
  - [Screenshots of the web application](#screenshots-of-the-web-application)
    - [Homepage](#homepage)
    - [NORMAL prediction](#normal-prediction)
    - [Low-grade glioma (LGG) prediction](#low-grade-glioma-lgg-prediction)
    - [High-grade glioma (HGG) prediction](#high-grade-glioma-hgg-prediction)

## Citation

Please see [CITATION.cff](CITATION.cff) for the full citation information.

### APA

```apa
Lie Uberg, L. (2022). Tumorclass.info (Version 0.1.0) [Computer software]. https://github.com/lewiuberg/tumorclass.info
```

### BibTex

```BibTex
@software{Lie_Uberg_Tumorclass_info_2022,
author = {Lie Uberg, Lewi},
license = {MIT},
month = {5},
title = {{Tumorclass.info}},
url = {https://github.com/lewiuberg/tumorclass.info},
version = {0.1.0},
year = {2022}
}
```

## Preprocessing

The preprocessing pipeline made for the original REMBRANDT dataset can be found in the [preprocessing](data/original_rembrandt/preprocessing.ipynb) notebook.

## Augmentation

The augmentation pipeline made for the manually labeled REMBRANDT dataset can be found in the [augmentation](notebooks/augmentation.ipynb) notebook.

## Model Training

The different model training pipelines can be found in the [notebooks](notebooks/) directory.

## User Interface

The runs from the [main.py](app/main.py) script can be found in the [app](app/) directory. This application utilizes the many of the files and subdirectories in project root directory. In order to run a local version of the application, clone this repository. After cloning, Poetry can be used to install dependencies. Since the `main.py` file is not located in the project root, you may need to export the python path using this command:

```bash
export PYTHONPATH=${PWD}/:${PWD}/app/
```

With a terminal open at the root of the project directory, run the following command to start the application:

```bash
poetry run python app/main.py
```

When the application is running, you can access it at the following URL http://localhost:8000

### Screenshots of the web application

#### Homepage

![home](/static/images/home.png "homepage")

#### NORMAL prediction

![normal](/static/images/normal.png "normal")

#### Low-grade glioma (LGG) prediction

![low-grade-glioma](/static/images/lgg.png "low-grade-glioma")

#### High-grade glioma (HGG) prediction

![high-grade-glioma](/static/images/hgg.png "high-grade-glioma")
