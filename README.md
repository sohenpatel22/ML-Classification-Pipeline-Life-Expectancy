# Life Expectancy Classification Project

## Overview
This project predicts whether a country's life expectancy is above or below the median using machine learning techniques.

The dataset contains health, economic, and social indicators from 193 countries (2000–2015).

---

## Objectives
- Perform data preprocessing and cleaning
- Convert regression problem into classification
- Train and evaluate KNN model
- Apply proper ML practices (train/test split, scaling, CV)

---

## Project Structure

life_expectancy_project/
│
├── data/
│   └── LifeExpectancy.csv
├── src/
│   ├── train.py
│   ├── preprocess.py
│   └── data_loader.py
├── README.md
└── .gitignore

---

## Setup

```bash
pip install pandas scikit-learn numpy