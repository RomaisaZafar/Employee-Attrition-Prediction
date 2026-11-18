import pandas as pd

data = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# Select Important Features

features = [
    "Age",
    "OverTime",
    "MonthlyIncome",
    "JobSatisfaction",
    "YearsAtCompany",
    "WorkLifeBalance",
    "EnvironmentSatisfaction",
    "DistanceFromHome"
]

X = data.drop("Attrition", axis=1)
y = data["Attrition"]


print("Features:", X.shape)
print("Target:", y.shape)

X = pd.get_dummies(X, drop_first=True, dtype=int)

print(X.head())
print("New Features Shape:", X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Model Training Completed!")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced'
)

dt_model.fit(X_train, y_train)

print("Decision Tree Training Completed!")

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

from sklearn.svm import SVC

svm_model = SVC(
    kernel='rbf',
    C=10,
    class_weight='balanced'
)

svm_model.fit(X_train, y_train)

print("SVM Training Completed!")

svm_scores = svm_model.decision_function(X_test)

svm_pred = ["Yes" if score > -0.2 else "No" for score in svm_scores]

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

print("\nClassification Report:")
print(classification_report(y_test, svm_pred))



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# 1. Confusion Matrices

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d')
plt.title("SVM Confusion Matrix")
plt.show()

sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.show()


# 2. Accuracy Comparison

plt.bar(
    ["Logistic Regression", "SVM", "Decision Tree"],
    [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, dt_pred)
    ]
)

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.show()


# 3. Precision, Recall and F1-score

metrics = ["Precision", "Recall", "F1-score"]

lr = [
    precision_score(y_test, y_pred, pos_label="Yes"),
    recall_score(y_test, y_pred, pos_label="Yes"),
    f1_score(y_test, y_pred, pos_label="Yes")
]

svm = [
    precision_score(y_test, svm_pred, pos_label="Yes"),
    recall_score(y_test, svm_pred, pos_label="Yes"),
    f1_score(y_test, svm_pred, pos_label="Yes")
]

dt = [
    precision_score(y_test, dt_pred, pos_label="Yes"),
    recall_score(y_test, dt_pred, pos_label="Yes"),
    f1_score(y_test, dt_pred, pos_label="Yes")
]

x = range(3)

plt.bar(x, lr, width=0.25, label="Logistic Regression")
plt.bar([i + 0.25 for i in x], svm, width=0.25, label="SVM")
plt.bar([i + 0.5 for i in x], dt, width=0.25, label="Decision Tree")

plt.xticks([i + 0.25 for i in x], metrics)
plt.title("Model Performance Comparison")
plt.legend()
plt.show()


# 4. Attrition Distribution

data["Attrition"].value_counts().plot(kind="bar")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()


import joblib

joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")

print("All Three Models, Scaler and Columns Saved Successfully!")

# Save Model Performance

results = {
    "Model": ["Logistic Regression", "SVM", "Decision Tree"],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, dt_pred)
    ],

    "Precision": [
        precision_score(y_test, y_pred, pos_label="Yes"),
        precision_score(y_test, svm_pred, pos_label="Yes"),
        precision_score(y_test, dt_pred, pos_label="Yes")
    ],

    "Recall": [
        recall_score(y_test, y_pred, pos_label="Yes"),
        recall_score(y_test, svm_pred, pos_label="Yes"),
        recall_score(y_test, dt_pred, pos_label="Yes")
    ],

    "F1-score": [
        f1_score(y_test, y_pred, pos_label="Yes"),
        f1_score(y_test, svm_pred, pos_label="Yes"),
        f1_score(y_test, dt_pred, pos_label="Yes")
    ]
}

pd.DataFrame(results).to_csv("model_results.csv", index=False) 