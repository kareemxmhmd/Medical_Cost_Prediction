# Medical Cost Prediction

## Overview
A machine learning project to predict medical insurance costs using patient demographics and health factors. The goal was to develop an accurate regression model that estimates insurance charges based on age, BMI, number of children, smoking status, and geographic region to support data-driven decision making in healthcare insurance.

## Technical Approach
Implemented a robust machine learning pipeline using scikit-learn:

- **Data Preprocessing**: 
  - Analyzed 1,338 patient records with comprehensive EDA
  - Applied StandardScaler to numeric features (age, BMI, children)
  - Used OneHotEncoder for categorical features (sex, smoker, region)
  
- **Model Architecture**: 
  - Implemented RandomForestRegressor with 100 estimators
  - Created scikit-learn Pipeline for seamless preprocessing and prediction
  - Achieved high predictive accuracy through cross-validation

- **Deployment**: 
  - Saved trained model as `model.pkl`
  - Developed `predict.py` for real-time cost predictions
  - Created `main.ipynb` documenting complete analysis workflow

## Results
- **High Accuracy**: Model provides reliable cost estimates for insurance pricing
- **Key Finding**: Smoking status shows strongest correlation with medical charges
- **Practical Value**: Enables insurance companies to assess risk and set premiums accurately
- **Complete Package**: End-to-end solution from data analysis to deployment