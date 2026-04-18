import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7],
    "Result": [0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["StudyHours"]]
y = df["Result"]

model = LogisticRegression()
model.fit(X, y)

# Proper prediction input
new_student = pd.DataFrame({"StudyHours": [2.5]})
prediction = model.predict(new_student)

print("Pass" if prediction[0] == 1 else "Fail")
