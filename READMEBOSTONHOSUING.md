# House Price Prediction using Machine Learning

## Project Overview

This project uses the **Boston Housing Dataset** to build a machine learning model that predicts the **median value of houses (medv)** in Boston. We utilize **Linear Regression** as the main model, along with several data preprocessing steps, including handling missing values, outlier detection, and feature scaling. Additionally, we explore the impact of regularization techniques and feature importance.

## Steps Followed:

### 1. **Data Loading and Initial Exploration**

- **Dataset**: The dataset is read using `pandas` from a CSV file.
- **Data Columns**: The dataset contains 14 columns, including features like crime rate, average number of rooms, and the proportion of residents below poverty.

### 2. **Data Preprocessing**

- **Handling Missing Values**: 
    - Checked for missing values and handled accordingly (removed rows with missing values, where necessary).
    - Columns with missing values (like `rm`) were handled by imputation or removal.
  
- **Outlier Detection and Removal**:
    - Identified outliers using the **Interquartile Range (IQR)** method.
    - Rows containing a high number of outliers (e.g., more than 3 outliers) were removed from the dataset.

- **Feature Scaling**:
    - Applied **Min-Max Scaling** to scale numerical features to a range between 0 and 1, ensuring uniformity and better performance during model training.

### 3. **Exploratory Data Analysis (EDA)**

- **Visualization**:
    - **Histograms**: For visualizing the distribution of each feature.
    - **Box Plots**: To identify the presence of outliers.
    - **Correlation Matrix**: Displayed as a heatmap to visualize the relationships between features and the target variable (`medv`).
    - **Scatter Plots**: To see the relationships between `medv` and other variables.
  
### 4. **Modeling**

- **Linear Regression**:
    - Split the data into training and test sets (80% train, 20% test).
    - Fit a **Linear Regression** model and evaluate its performance using Mean Squared Error (MSE) and R² score.
    - A **scatter plot** of actual vs predicted values was plotted for visual evaluation.
    - **Residual plot** was analyzed to check for homoscedasticity and model assumptions.

- **Regularization (Ridge Regression)**:
    - To reduce overfitting, Ridge Regression was applied using **L2 regularization**. The model’s performance was evaluated with MSE and R².

- **Cross-validation and Hyperparameter Tuning**:
    - **GridSearchCV** was used to tune hyperparameters for models like **Random Forest** to find the best configuration for better predictions.
  
### 5. **Model Evaluation**

- The performance of the model was assessed using:
    - **Mean Squared Error (MSE)**: Indicates the average squared difference between predicted and actual values.
    - **R² score**: Represents how well the model explains the variance in the target variable.

### 6. **Model Interpretability**

- **Feature Importance**: 
    - Used **Random Forest** to identify the most important features that affect the target variable (`medv`).
    - A bar chart of feature importances was plotted.

- **Outlier Detection**: 
    - Used the **IQR method** to detect and remove outliers.
  
- **Residuals Analysis**:
    - A residual plot was generated to ensure that the model assumptions (such as homoscedasticity) were not violated.

## Requirements

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- mlxtend
- joblib

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels mlxtend joblib
  
