**Predicting House Prices Using Regression: A Hands-On Exploration**

---

### 1. Introduction

In this project, I explored the Kaggle House Prices competition dataset, where the main goal is to predict house prices based on various features of the properties. This dataset includes a mix of numerical and categorical variables, such as overall quality, year built, living area, and neighborhood. My aim was to apply regression techniques to build predictive models, evaluate their performance, and gain experience working with real-world data.

---

### 2. What is Regression?

Regression is a type of supervised machine learning used to model the relationship between a dependent variable (target) and one or more independent variables (features). The main objective is to predict a continuous value.

#### Linear Regression

Linear regression assumes a linear relationship between the input features and the target. The model tries to fit a line (or hyperplane in higher dimensions) that minimizes the difference between the predicted and actual values.

Mathematically:
\(y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \varepsilon\)

Where:

- \(y\): predicted price
- \(\beta_0\): intercept
- \(\beta_i\): coefficient for feature \(x_i\)
- \(\varepsilon\): error term

The coefficients are learned using the Least Squares method, which minimizes the sum of squared errors between the predicted and actual values.

---

### 3. Experiment 1: Baseline Model

#### Data Understanding

To start, I loaded the `train.csv` dataset and examined it using `.info()` and `.describe()` methods. I also visualized feature correlations using a heatmap. Features like `GrLivArea`, `TotalBsmtSF`, and `OverallQual` showed strong correlations with the target `SalePrice`.

#### Pre-processing

- Dropped rows with missing target values.
- Imputed missing numerical features with the median.
- Applied label encoding for ordinal categorical features and one-hot encoding for nominal ones.
- Selected initial features: `GrLivArea`, `OverallQual`, `YearBuilt`, `GarageCars`, `TotalBsmtSF`.

#### Modeling

I used `LinearRegression` from scikit-learn to train the model on the training data.

#### Evaluation

I evaluated the model using RMSE (Root Mean Squared Error). The baseline model yielded an RMSE of approximately 35,000. While not perfect, it provided a reference point for further experimentation.

---

### 4. Experiment 2: Feature Expansion and Outlier Removal

In this experiment, I expanded the feature set to include `1stFlrSF`, `GarageArea`, and `BsmtFinSF1`. I also identified and removed outliers, particularly houses with extremely high `GrLivArea` but relatively low `SalePrice`.

After retraining the linear regression model, the RMSE dropped to around 31,000, indicating improved performance.

---

### 5. Experiment 3: Regularization with Ridge Regression

To address potential multicollinearity and overfitting, I implemented Ridge regression. This technique adds a penalty to the size of coefficients:

\(\text{Loss} = \text{RSS} + \alpha \sum_{i=1}^{n} \beta_i^2\)

After tuning the alpha parameter, the Ridge model with `alpha=10` achieved an RMSE of approximately 29,500, improving over the basic linear model.

---

### 6. Impact Section

Predictive models in real estate can greatly influence buying/selling decisions, city planning, and investment strategies. However, they can also perpetuate biases if the training data reflects social inequalities (e.g., neighborhood-based pricing that reflects historical segregation). Ethical considerations are critical when deploying such models.

---

### 7. Conclusion

From this project, I learned:

- How data cleaning and feature engineering greatly influence model performance.
- Linear regression is a solid baseline, but regularization helps in improving generalization.
- Visualizations and exploratory data analysis are essential steps in the modeling process.

---

### 8. References

- Kaggle House Prices Competition: [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Python Data Science Handbook by Jake VanderPlas

---

### 9. Code

All code is available in the accompanying Jupyter notebook on my GitHub:

https://github.com/Legend700/Datacharlotte/blob/main/house_price_regression_project.ipynb