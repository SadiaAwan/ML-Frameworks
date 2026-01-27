# Task 3: Framework comparison in code
# TODO: Using scikit-learn, load the iris dataset
# TODO: Train a LogisticRegression model
# TODO: Train a tiny MLP (MLPClassifier) on the same data
# TODO: Compare accuracy and write 3-5 comments in code about:
# - speed
# - API ergonomics
# - when you would pick each approach

# 0. Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time 

# 1) Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split(Keep class proportions stable with stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Logistic Regression pipline
# NOTE: Scaling isn't always required for LogisticRegression, but it often helps optimization.
logreg = Pipeline(
    steps=[
        ("Scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

t0 = time.perf_counter()
logreg.fit(X_train, y_train)
logreg_train_time = time.perf_counter() - t0

y_pred_lr = logreg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# 4) Tiny MLP pipeline
# For MLSPs, scaling is basically mandatory for stable traning
mlp = Pipeline(
    steps =[
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42,
            early_stopping=True,
        )),
    ]
)
t0 = time.perf_counter()
mlp.fit(X_train, y_train)
mlp_train_time = time.perf_counter() - t0

y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

print("=== Result on Iris (test set) ===")
print(f"LogisticRegression  accuracy={acc_lr:.3f}, train_time={logreg_train_time*1000:.1f} ms")
print(f"MLPClassifier  accuracy={acc_mlp:.3f}, train_time={mlp_train_time*1000:.1f} ms")


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

# 4) When to pick MLPClassifier:
# When you suspect non-linear decision boundaries AND you have enough data, plus willingness to tune.
# On tiny datasets it can overfit or be finicky, but it can beat linear models when non-linear structure is real.
#
# 5) Practical MLOps angle:
# LogisticRegression is easier to monitor/debug in production (simpler behavior). MLPs may need more care:
# calibration checks, drift sensitivity, and more attention to training reproducibility/hyperparameter tracking.
# -------------------------------

