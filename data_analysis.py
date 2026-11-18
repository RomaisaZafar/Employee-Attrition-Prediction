import pandas as pd

data = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nMissing Values:", data.isnull().sum().sum())

print("\nAttrition Distribution:")
print(data["Attrition"].value_counts())

print("\nDuplicate Rows:", data.duplicated().sum())

import matplotlib.pyplot as plt

data["Attrition"].value_counts().plot(kind="bar")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.show()
import seaborn as sns
# 1. Attrition by Department

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="Department", hue="Attrition")

plt.title("Employee Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()

# 2. Attrition by Overtime

plt.figure(figsize=(7, 5))

sns.countplot(data=data, x="OverTime", hue="Attrition")

plt.title("Employee Attrition by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()

# 3. Monthly Income vs Attrition

plt.figure(figsize=(7, 5))

sns.barplot(data=data, x="Attrition", y="MonthlyIncome")

plt.title("Average Monthly Income by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Average Monthly Income")

plt.tight_layout()
plt.show()

# 4. Years at Company vs Attrition

plt.figure(figsize=(7, 5))

sns.barplot(data=data, x="Attrition", y="YearsAtCompany")

plt.title("Average Years at Company by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Average Years")

plt.tight_layout()
plt.show()

# Drop Unnecessary Columns

data = data.drop(["EmployeeCount", "EmployeeNumber",
                  "Over18", "StandardHours"], axis=1)

print("\nDataset After Removing Unnecessary Columns:")
print(data.head())