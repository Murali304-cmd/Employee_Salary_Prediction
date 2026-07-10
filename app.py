import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.stButton>button{
    width:100%;
    background-color:#2563eb;
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load Model & Scaler
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("random_forest_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# -----------------------------------------------------
# Dictionaries (Modify if required)
# -----------------------------------------------------
gender_dict = {
    "Female": 0,
    "Male": 1
}

department_dict = {
    "Finance": 0,
    "HR": 1,
    "IT": 2,
    "Marketing": 3,
    "Sales": 4
}

job_dict = {
    "Analyst": 0,
    "Developer": 1,
    "Engineer": 2,
    "Manager": 3,
    "Consultant": 4
}

education_dict = {
    "Bachelor": 0,
    "Master": 1,
    "PhD": 2
}

location_dict = {
    "Austin": 0,
    "Chicago": 1,
    "Dallas": 2,
    "New York": 3
}

# -----------------------------------------------------
# Title
# -----------------------------------------------------
st.title("💼 Employee Salary Prediction System")
st.write("Predict employee salary using Machine Learning.")

# -----------------------------------------------------
# Input
# -----------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    gender = st.selectbox(
        "Gender",
        list(gender_dict.keys())
    )

    department = st.selectbox(
        "Department",
        list(department_dict.keys())
    )

with col2:

    job = st.selectbox(
        "Job Title",
        list(job_dict.keys())
    )

    experience = st.slider(
        "Experience (Years)",
        0,
        40,
        5
    )

    education = st.selectbox(
        "Education",
        list(education_dict.keys())
    )

    location = st.selectbox(
        "Location",
        list(location_dict.keys())
    )

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
if st.button("💰 Predict Salary"):

    x = np.array([[
        age,
        gender_dict[gender],
        department_dict[department],
        job_dict[job],
        experience,
        education_dict[education],
        location_dict[location]
    ]])

    x = scaler.transform(x)

    prediction = float(model.predict(x)[0])

    st.success(f"### 💰 Predicted Salary : {prediction:,.2f}")

    # Salary Level
    if prediction < 40000:
        level = "🟢 Entry Level"
    elif prediction < 70000:
        level = "🔵 Mid Level"
    elif prediction < 100000:
        level = "🟠 Senior Level"
    else:
        level = "👑 Executive Level"

    # Next Year Salary
    next_year = prediction * 1.10

    colA, colB, colC = st.columns(3)

    colA.metric("Current Salary", f"{prediction:,.2f}")
    colB.metric("Next Year Salary", f"{next_year:,.2f}", "10%")
    colC.metric("Career Level", level)

    # Employee Summary
    st.subheader("👨 Employee Summary")

    summary = pd.DataFrame({
        "Field": [
            "Age",
            "Gender",
            "Department",
            "Job Title",
            "Experience",
            "Education",
            "Location"
        ],
        "Value": [
            age,
            gender,
            department,
            job,
            f"{experience} Years",
            education,
            location
        ]
    })

    st.dataframe(summary, use_container_width=True)

    # Future Salary Prediction
    st.subheader("📈 Future Salary Prediction")

    salary = prediction

    years = []
    salaries = []

    for i in range(1, 6):
        salary *= 1.10
        years.append(f"Year {i}")
        salaries.append(salary)

    future_df = pd.DataFrame({
        "Year": years,
        "Estimated Salary": salaries
    })

    st.dataframe(future_df, use_container_width=True)

    # Salary Growth Chart
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(years, salaries, marker="o", linewidth=3)

    ax.set_title("Salary Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Salary ")

    st.pyplot(fig)

    # Career Progress
    st.subheader("⭐ Career Progress")

    progress = min(int((experience / 20) * 100), 100)

    st.progress(progress)

    # Quotes
    quotes = [

        "🚀 Keep learning to unlock higher salaries.",

        "💼 Skills create opportunities.",

        "📈 Consistency drives career growth.",

        "🌟 Invest in your knowledge.",

        "🏆 Great careers are built one skill at a time.",

        "💡 Every certification increases your value.",

        "🔥 Learn today, lead tomorrow.",

        "🎯 Hard work beats talent when talent doesn't work hard.",

        "📚 Never stop learning.",

        "🚀 Success comes from continuous improvement."

    ]

    st.info(random.choice(quotes))

    # CSV Download
    csv = future_df.to_csv(index=False).encode()

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "salary_prediction.csv",
        "text/csv"
    )

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.success("Random Forest Regressor")

st.sidebar.write("Scaler : StandardScaler")

st.sidebar.write("Target : Salary")

st.sidebar.markdown("---")

st.sidebar.write("### Features Used")

st.sidebar.write("""
- Age
- Gender
- Department
- Job Title
- Experience
- Education
- Location
""")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")

st.caption(
    "💼 Employee Salary Prediction System | "
    "Machine Learning Project | "
    "Developed using Streamlit"
)
