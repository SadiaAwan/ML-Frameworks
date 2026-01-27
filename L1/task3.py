# Task 3: Framework comparison in code
# TODO: Using scikit-learn, load the iris dataset
# TODO: Train a LogisticRegression model
# TODO: Train a tiny MLP (MLPClassifier) on the same data
# TODO: Compare accuracy and write 3-5 comments in code about:
# - speed
# - API ergonomics
# - when you would pick each approach

# 0. Imports
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset

# Load the built-in Iris dataset from scikit-learn
X, y = load_iris(return_X_y=True)

print("Feature shape:", X.shape)
print("Feature shape:", y.shape)

# 2. Train test split

# Split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 3. Train Logistic Regression

# Create Logistic Regression model
log_reg = LogisticRegression(max_iter=200)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on test data
y_pred_lr = log_reg.predict(X_test)

# Compute accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression accuracy:", acc_lr)

# 4. Train a tiny Neural Network (MLP)

# Create a tiny MLP with one hidden layer of 20 neurons
mlp = MLPClassifier(
    hidden_layer_sizes=(20,),
    max_iter=500,
    random_state=42,
)

# Train the neural network
mlp.fit(X_train, y_train)

# Predict on test data
y_pred_mlp = mlp.predict(X_test)

# Compute accuracy
acc_mlp = accuracy_score(y_test, y_pred_mlp)

print("MLP accuracy:", acc_mlp)

# 5. Write comparison comments IN CODE


# -------------------------------
# Comparison Notes:
#
# 1)Speed:
# - LogisticRegression trains extremely fast on small tabular datasets.
# - MLPClassifier is slower because it does iterative gradient descent.
#
# 2)API Ergonomics:
# - Both use the same scikit-learn API: fit(), predict(), score().
# - LogisticRegression has fewer hyperparameters and is simpler to tune.
#
# 3) When to pick each:
# - Pick LogisticRegression for simple, interpretable baselines.
# - Pick MLP when the data has nonlinear patterns a linear model cannot capture.
# - Always try LogisticRegression first as a benchmark.
# -------------------------------

