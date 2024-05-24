# Ensemble Methods with scikit-learn

This repository contains a Jupyter Notebook that introduces the basics of ensemble methods and demonstrates their implementation using scikit-learn. Through practical examples, the notebook shows how to prepare data, build individual models, and construct ensemble models to enhance predictive performance.

## Overview

### Ensemble Methods
Ensemble methods combine multiple individual models to improve overall performance and robustness. This notebook focuses on two popular ensemble techniques:
- **Decision Trees**: A basic building block for many ensemble methods.
- **Random Forests**: An ensemble of decision trees that reduces overfitting and increases accuracy.

## Notebook Content

### Step 1: Preparing the Data
The first step involves loading the dataset, defining the features and target variable, and splitting the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/SDG_15_Life_on_Land_Dataset.csv')

# Define features and target
X = data.drop('BiodiversityHealthIndex', axis=1)
y = data['BiodiversityHealthIndex']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 2: Building Individual Models
A decision tree regressor is built and evaluated using mean squared error (MSE) to establish a baseline performance.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Initialise and train the decision tree
tree_model = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_model.fit(X_train, y_train)

# Predict and evaluate
tree_predictions = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)
print(f"Decision Tree MSE: {tree_mse}")

# Output: Decision Tree MSE: 0.08696874034772016
```

### Step 3: Building an Ensemble Model
A random forest regressor, which is an ensemble of decision trees, is built and evaluated. The ensemble model is expected to outperform the individual decision tree by reducing overfitting and leveraging the strength of multiple trees.

```python
from sklearn.ensemble import RandomForestRegressor

# Initialise and train the random forest
forest_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
forest_model.fit(X_train, y_train)

# Predict and evaluate
forest_predictions = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_predictions)
print(f"Random Forest MSE: {forest_mse}")

# Output: Random Forest MSE: 0.0858280816891359
```

### Results
The notebook demonstrates the effectiveness of ensemble methods by comparing the performance of a single decision tree to that of a random forest. The random forest, with its ensemble approach, achieves a lower MSE, highlighting the benefits of combining multiple models.

## Usage
To run this notebook, you need to have Python and the necessary libraries installed. Clone this repository and open the notebook to follow along with the examples and implement your own ensemble models.

```bash
git clone https://github.com/yourusername/Ensemble-Methods-Notebook.git
cd Ensemble-Methods-Notebook
jupyter notebook
```

## Conclusion
This notebook provides a clear and practical introduction to ensemble methods, demonstrating how to build and evaluate both individual and ensemble models. It serves as a valuable resource for data scientists and machine learning practitioners looking to improve their models using ensemble techniques.

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to enhance this repository.
