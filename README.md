# Glucose prediction and insulin recommendation

## Problem description

This project aims to predict glucose levels and recommend insulin doses for patients with type 2 diabetes. The data used in this project is collected from the Zhongshan Hospital Affiliated to Fudan University. The data includes the patient's time series features including: glucose levels, insulin doses, and other drag usages, and static features including: age, gender, height, weight, BMI, and other clinical features.

In next step, we will incorporate the patient's data from other hospitals to improve the model's generalization ability by using federated learning.

## How to use

### Dependencies

python                    3.8.16
wandb                     0.13.10
pytorch                   1.8.0
transformers              4.24.0
torchvision               0.9.0  
pandas                    1.4.2
numpy                     1.23.5

### Data Preparation for Type-2 Diabetes Dataset

The dataset contains both basic information and time-series data from type-2 diabetes patients:

- Basic Information
  - Patient Details: ID, age, gender, height, weight, BMI.
  - Chemical Indicators: 33 chemical indicators.
  - Related Diseases: 15 highly related diseases.

- Time-Series Data
  - Glucose Levels: Measurements taken multiple times daily (before/after meals and bedtime).
  - Insulin Doses: Categorized into basal, premix, and shot.
  - Other Drug Usages: Categorized into 15 types based on the principal ingredient.

- Feature Engineering
  - Handling Abnormal Values: Replace abnormal values with the median of the corresponding feature.
  - Handling Missing Values: Replace missing values for continuous variables such as age, height, and weight with the  median value. Replace missing values for chemical indicators with 0.
  - Normalization:
    - For Continuous Indicators: Normalize based on the recommended range [ \text{min}, \text{max}].
      - Within Range: Set value to 0.
      - Below Range: (x - \text{min}) / (\text{max} - \text{min}).
      - Above Range: (x - \text{max}) / (\text{max} - \text{min}).
      - This approach emphasizes abnormalities in the data.
    - For Qualitative Indicators:
      - Use scalar representation [0, 1, \ldots], where 0 indicates normality, and other values represent different levels of abnormality.
  - Categorization:
    - Insulins: Grouped into basal, premix, and shot.
    - Other drugs: Categorized into 15 types based on principal ingredients.
  - Time-Series Data Preparation:
    - Retain data only for patients with hospital stays longer than 2 days.
    - Use the first 8 days of time-series data for training and prediction.
    - Save each patient’s data as a separate CSV file named after their ID. Each CSV row represents a patient’s blood glucose and medication details at a specific time point. Expected daily records: 7 (before/after meals and bedtime), although missing values are common.

### Code structure

- 'glu_dataset.py': the data processing script.
- 'model.py': the transformer architecture.
- 'train.py': the training script.
- 'run.sh': the bash script to start multiple runs.

### Training example

```bash
train.py --n_layer=1 --learning_rate=0.03 --dropout=0.1 --batch_size=64
```
