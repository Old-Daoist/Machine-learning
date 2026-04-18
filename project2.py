import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7],
    "Result": [0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["StudyHours"]]
y = df["Result"]

# Train model
model = LogisticRegression()
model.fit(X, y)
print("Weight (w):", model.coef_)
print("Bias (b):", model.intercept_)

# Generate continuous values for plotting curve
x_values = np.linspace(0, 8, 100).reshape(-1, 1)
y_prob = model.predict_proba(x_values)[:, 1]  # Probability of Pass

# Plot data points
plt.scatter(df["StudyHours"], df["Result"], label="Actual Data")

# Plot sigmoid curve
plt.plot(x_values, y_prob, label="Logistic Regression Curve")

# Decision boundary (probability = 0.5)
plt.axhline(0.5, linestyle="--", label="Decision Boundary (0.5)")

plt.xlabel("Study Hours")
plt.ylabel("Pass Probability")
plt.title("Logistic Regression: Study Hours vs Pass Probability")
plt.legend()
plt.grid(True)
plt.show()
