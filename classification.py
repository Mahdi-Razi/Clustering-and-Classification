import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Extract features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification with Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)  # Increase max_iter
logistic_regression.fit(X_train, y_train)

# Predict labels on the test set
y_pred = logistic_regression.predict(X_test)

# Evaluate the Logistic Regressaion model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='viridis', s=10)
plt.title('Scatter Plot of Classes (Predicted)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()