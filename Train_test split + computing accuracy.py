# Import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# Import the module for splitting the data
from sklearn.model_selection import train_test_split
churn_df = pd.read_csv(r"D:\DataCamp_ADS\Supervised Learning with scikit-learn\Codes\telecom_churn_clean.csv")
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test,y_test))