# ML Standing vs Sitting

Mini machine learning project to classify whether a person is **standing** or **sitting** using smartphone sensor data.

## Objective

Build a simple and reproducible ML pipeline for binary human activity classification using the **UCI Human Activity Recognition Using Smartphones** dataset.

## Dataset

Source dataset:
- UCI Human Activity Recognition Using Smartphones

Original classes include:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

For this project, we will focus only on:
- **SITTING**
- **STANDING**

## Project structure

```text
ml-standing-vs-sitting/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── .gitignore