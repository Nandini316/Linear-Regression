# Linear-Regression
Understanding Linear Regression 
# Linear Regression from Scratch

This project implements a simple linear regression model from scratch using Python and demonstrates its usage on a synthetic dataset. The goal is to predict continuous values by learning a linear relationship between input features and the target variable.

### Key Libraries Used:
- **NumPy**: For numerical operations and handling arrays.
- **scikit-learn**: For dataset generation and splitting.
- **matplotlib**: For data visualization.

### Files in the Repository:
1. **`lin_reg_test.py`**:
   - This file contains the implementation of the `LinearRegressions` class, which simulates a linear regression model with gradient descent optimization.

2. **Main Script**:
   - The main Python script generates a dataset, trains the linear regression model, and evaluates it using mean squared error (MSE).

### Features:
- **Custom Linear Regression Class**:
  - The `LinearRegressions` class implements the following methods:
    - `fit(X, y)`: Trains the model by updating the weights and bias using gradient descent.
    - `predict(X)`: Predicts the target values for a given input dataset using the learned weights and bias.
  
- **Synthetic Dataset**:
  - A synthetic regression dataset is created using `make_regression` from `scikit-learn` with noise for added realism.

- **Model Training**:
  - The dataset is split into training and test sets using `train_test_split` from `scikit-learn`, and the `LinearRegressions` model is trained on the training data.

- **Model Evaluation**:
  - The performance of the model is evaluated using Mean Squared Error (MSE), a common metric for regression tasks, and the model's predictions are visualized.

### Visualization:
- The dataset and the linear regression predictions are plotted using `matplotlib`. The training data, test data, and the predicted regression line are clearly visualized for better understanding.


3. **Expected Output**:
   The script will output the mean squared error (MSE) of the model and display a plot showing the training and test data points along with the regression line.
