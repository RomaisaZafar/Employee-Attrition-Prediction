import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# Custom Styling

# Custom Styling
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f4f7fb;
    color: #1f2937;
}

/* Headings */
h1, h2, h3, h4 {
    color: #17365d !important;
    font-weight: 700;
}

/* Normal Text */
p, label, span, li {
    color: #334155;
}

/* Widget Labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #17365d !important;
}

/* Input Fields */
[data-baseweb="input"] input,
[data-baseweb="select"] > div {
    background-color: white !important;
    color: #1f2937 !important;
    border-color: #cbd5e1 !important;
}

/* Selectbox Text */
[data-baseweb="select"] * {
    color: #1f2937 !important;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background-color: #17365d;
    color: white !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: #17365d !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #dce5ef;
}

/* White Input Fields */
[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}

[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

/* White Dropdown Fields */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
}

[data-baseweb="select"] input {
    color: #1f2937 !important;
}

/* Number Input Controls */
[data-testid="stNumberInput"] button {
    background-color: #ffffff !important;
    color: #17365d !important;
}

/* Predict Button */
.stButton button,
.stFormSubmitButton button {
    background-color: #17365d !important;
    color: #ffffff !important;
}

.stButton button p,
.stFormSubmitButton button p,
.stButton button span,
.stFormSubmitButton button span {
    color: #ffffff !important;
}


</style>
""", unsafe_allow_html=True)

# Load Models
model = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_columns.pkl")

# Load Dataset
data = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
results = pd.read_csv("model_results.csv")

# Header
st.title("Employee Attrition Prediction")
st.write("An Employee Analytics and Attrition Prediction System")
st.divider()

# Tabs
prediction_tab, analysis_tab, performance_tab = st.tabs(
    ["Employee Prediction", "Data Analysis", "Model Performance"]
)

# ---------------- PREDICTION TAB ----------------

with prediction_tab:

    st.header("Employee Information")
    st.write("Enter employee details to predict the possibility of attrition.")

    with st.form("employee_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 60, 30)
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            income = st.number_input("Monthly Income", 1000, 20000, 5000)
            satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            years = st.number_input("Years at Company", 0, 40, 3)

        with col2:
            worklife = st.selectbox("Work Life Balance", [1, 2, 3, 4])
            environment = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            distance = st.number_input("Distance From Home", 1, 29, 5)
            role = st.number_input("Years in Current Role", 0, 18, 2)
            selected_model = st.selectbox(
                "Select Model",
                ["Logistic Regression", "Decision Tree", "SVM"]
            )

        submit = st.form_submit_button("Predict Attrition")

    if submit:

        employee = data.drop("Attrition", axis=1).copy()

        for column in employee.columns:
            if pd.api.types.is_numeric_dtype(employee[column]):
                employee[column] = employee[column].median()
            else:
                employee[column] = employee[column].mode()[0]

        employee = employee.iloc[[0]]

        employee["Age"] = age
        employee["OverTime"] = overtime
        employee["MonthlyIncome"] = income
        employee["JobSatisfaction"] = satisfaction
        employee["YearsAtCompany"] = years
        employee["WorkLifeBalance"] = worklife
        employee["EnvironmentSatisfaction"] = environment
        employee["DistanceFromHome"] = distance
        employee["YearsInCurrentRole"] = role

        employee = pd.get_dummies(employee, drop_first=True, dtype=int)
        employee = employee.reindex(columns=columns, fill_value=0)

        employee = scaler.transform(employee)

        if selected_model == "Logistic Regression":
            result = model.predict(employee)[0]

        elif selected_model == "Decision Tree":
            result = dt_model.predict(employee)[0]

        else:
            score = svm_model.decision_function(employee)[0]
            result = "Yes" if score > -0.2 else "No"

        st.divider()
        st.subheader("Prediction Result")

        if result == "Yes":
            st.error("Employee is likely to leave the company.")
        else:
            st.success("Employee is likely to stay with the company.")

        st.metric("Selected Model", selected_model)

        st.caption(
            "Note: Other employee features are automatically filled "
            "using median/mode values from the dataset."
        )

# ---------------- DATA ANALYSIS TAB ----------------

with analysis_tab:

    st.header("Employee Data Analysis")
    st.write("Explore employee attrition patterns through visualizations.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees", len(data))
    col2.metric("Employees Left", (data["Attrition"] == "Yes").sum())
    col3.metric(
        "Attrition Rate",
        f"{(data['Attrition'] == 'Yes').mean() * 100:.1f}%"
    )

    st.divider()

    # 1. Attrition Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(3.5, 2))
        data["Attrition"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Attrition")
        ax.set_ylabel("Employees")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # 2. Attrition by Department
    with col2:
        st.subheader("Attrition by Department")
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        sns.countplot(data=data, x="Department", hue="Attrition", ax=ax)

        ax.set_xlabel("Department")
        ax.set_ylabel("Employees")
        plt.xticks(rotation=20, ha="right")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # 3. Attrition by Overtime
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition by Overtime")
        fig, ax = plt.subplots(figsize=(3.5, 2))
        sns.countplot(data=data, x="OverTime", hue="Attrition", ax=ax)
        ax.set_xlabel("OverTime")
        ax.set_ylabel("Employees")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # 4. Monthly Income vs Attrition
    with col2:
        st.subheader("Monthly Income vs Attrition")
        fig, ax = plt.subplots(figsize=(3.5, 2))
        sns.barplot(data=data, x="Attrition", y="MonthlyIncome", ax=ax)
        ax.set_xlabel("Attrition")
        ax.set_ylabel("Average Monthly Income")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # 5. Years at Company vs Attrition
    st.subheader("Years at Company vs Attrition")
    fig, ax = plt.subplots(figsize=(3.5, 2))
    sns.barplot(data=data, x="Attrition", y="YearsAtCompany", ax=ax)
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Average Years at Company")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---------------- MODEL PERFORMANCE TAB ----------------

with performance_tab:

    st.header("Model Performance Comparison")
    st.write("Comparison of the three trained machine learning models.")

    st.subheader("Performance Metrics")
    st.dataframe(results, use_container_width=True)

    st.subheader("Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(3.5, 2))
    sns.barplot(data=results, x="Model", y="Accuracy", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=15)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("Precision, Recall and F1-score")

    metrics = results.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(3.5, 2))

    sns.barplot(
        data=metrics,
        x="Metric",
        y="Score",
        hue="Model",
        ax=ax,
        width=0.6
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=3)
    ax.set_xlabel("", fontsize=3)
    ax.tick_params(axis="both", labelsize=3)
    ax.legend(fontsize=6, title_fontsize=3)

    plt.tight_layout(pad=0.5)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)