from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
# PlayTennis dataset
X = [
    ["Sunny", "Hot", "High", "Weak"],
    ["Sunny", "Hot", "High", "Strong"],
    ["Overcast", "Hot", "High", "Weak"],
    ["Rain", "Mild", "High", "Weak"],
    ["Rain", "Cool", "Normal", "Weak"],
    ["Rain", "Cool", "Normal", "Strong"],
    ["Overcast", "Cool", "Normal", "Strong"],
    ["Sunny", "Mild", "High", "Weak"],
    ["Sunny", "Cool", "Normal", "Weak"],
    ["Rain", "Mild", "Normal", "Weak"],
    ["Sunny", "Mild", "Normal", "Strong"],
    ["Overcast", "Mild", "High", "Strong"],
    ["Overcast", "Hot", "Normal", "Weak"],
    ["Rain", "Mild", "High", "Strong"]
]
Y = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
     "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
encoders = []
X_encoded = []

for col in range(4):
    le = LabelEncoder()
    column_values = [row[col] for row in X]
    encoded_column = le.fit_transform(column_values)
    encoders.append(le)
    X_encoded.append(encoded_column)

X_encoded = np.array(list(zip(*X_encoded)))

target_encoder = LabelEncoder()
Y_encoded = target_encoder.fit_transform(Y)
loo = LeaveOneOut()
loocv_predictions = []
loocv_actual = []
correct_count = 0

print("Decision Tree Leave-One-Out Cross Validation Results:")
print("---------------------------------------")

for i, (train_index, test_index) in enumerate(loo.split(X_encoded), start=1):
    X_train, X_test = X_encoded[train_index], X_encoded[test_index]
    y_train, y_test = Y_encoded[train_index], Y_encoded[test_index]

    dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt.fit(X_train, y_train)

    pred = dt.predict(X_test)

    loocv_predictions.append(pred[0])
    loocv_actual.append(y_test[0])

    if pred[0] == y_test[0]:
        correct_count += 1
        result = "Correct"
    else:
        result = "Wrong"

    print(f"Test {i}: Actual = {y_test[0]}, Predicted = {pred[0]} --> {result}")

loocv_accuracy = correct_count / len(Y_encoded)

print("---------------------------------------")
print("Correct Predictions:", correct_count, "out of", len(Y_encoded))
print("Decision Tree Accuracy:", loocv_accuracy)
print("\nClassification Report:")
print(classification_report(
    loocv_actual,
    loocv_predictions,
    target_names=["No", "Yes"]
))

# =========================
# 3. Tree Visualization
# =========================

# Train final model on the full dataset only for visualization
dt_final = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_final.fit(X_encoded, Y_encoded)

plt.figure(figsize=(5, 10))
plot_tree(
    dt_final,
    feature_names=["Outlook", "Temperature", "Humidity", "Wind"],
    class_names=["No", "Yes"],
    filled=True
)

plt.title("Decision Tree - PlayTennis Dataset")
plt.show()